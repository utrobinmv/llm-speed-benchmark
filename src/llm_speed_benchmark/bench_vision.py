#!/usr/bin/env python3
"""
llm_speed_benchmark/bench_vision.py

Бенчмарк скорости vision/video-модели (мультимодальная LLM) через OpenAI-compatible API.

Отправляет модели изображения и/или видео с запросом описания, измеряет:
  - TTFT (Time To First Token)
  - Скорость генерации (токенов/сек)
  - Мгновенную скорость
  - Общее время обработки

Три режима:
  - **image-only** (max_videos=0, max_images>0) — только изображения
  - **video-only** (max_videos>0, max_images=0) — только видео
  - **mixed** (max_videos>0 И max_images>0) — и изображения, и видео в одном запросе

Поддерживает N параллельных воркеров (multiprocessing) с Rich Live-таблицей.

Использование:
  bench_vision
  bench_vision --workers 4 --duration 120
  bench_vision --images ~/workspace/data/benchmark_images/ --max-images 4
  bench_vision --videos ~/workspace/data/test_videos/ --max-videos 4
  bench_vision --max-videos 2 --max-images 3   # mixed mode
"""

from __future__ import annotations

import os
import sys
import time
import signal
import argparse
import traceback
from multiprocessing import Process, Queue, Event
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from rich.live import Live
from rich.table import Table

from .utils import (
    API_KEY,
    BASE_URL,
    MODEL,
    format_time,
)
from .cli_common import add_common_args, apply_config
from .streaming import StreamSession
from .image_utils import (
    DEFAULT_PROMPTS,
    DEFAULT_VIDEO_PROMPTS,
    build_mixed_vision_message,
    build_vision_message,
    build_video_message,
    discover_images,
    discover_videos,
    generate_test_images,
    generate_test_videos,
    sawtooth_image_count,
)
from .live_table import BaseLiveTable
from .worker_common import make_time_sender, make_on_chunk_callback, send_stats, send_error_stats


# ---------------------------------------------------------------------------"
# Worker
# ---------------------------------------------------------------------------"

def _worker(  # type: ignore[reportInvalidTypeForm, valid-type]
    worker_id: int,
    q: "Queue",  # type: ignore[reportInvalidTypeForm, valid-type]
    start_event: "Event",  # type: ignore[reportInvalidTypeForm, valid-type]
    image_paths: List[str],
    video_paths: List[str],
    duration: Optional[int],
    prompts: List[str],
    max_images: int,
    max_videos: int,
    skip_errors: bool = False,
) -> None:
    """Worker: отправляет модели запросы с изображениями и/или видео.

    В image-only режиме (max_images>0, max_videos=0):
      - цикл по изображениям, пилообразный паттерн количества

    В video-only режиме (max_videos>0, max_images=0):
      - цикл по видео, пилообразный паттерн количества

    В mixed режиме (max_images>0 И max_videos>0):
      - в каждом запросе И изображения, И видео
      - независимые пилообразные паттерны для каждого типа
      - первый элемент каждого типа показывается в колонках Image/Video

    Args:
        worker_id: ID воркера.
        q: Очередь сообщений.
        start_event: Событие синхронизации старта.
        image_paths: Пути к изображениям.
        video_paths: Пути к видео.
        duration: Длительность в секундах (None = без лимита).
        prompts: Промпты для запросов.
        max_images: Максимум изображений в запросе (0 = без изображений).
        max_videos: Максимум видео в запросе (0 = без видео).
        skip_errors: Продолжать после ошибки.
    """
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)
        session = StreamSession(client)

        mixed_mode = max_images > 0 and max_videos > 0
        has_images = len(image_paths) > 0 and max_images > 0
        has_videos = len(video_paths) > 0 and max_videos > 0

        q.put({"type": "start", "id": worker_id,
               "media_img": len(image_paths),
               "media_vid": len(video_paths)})

        start_time = time.time()
        call_count = 0
        total_gen = 0
        total_chunks = 0
        total_ttft = 0.0
        ttft_count = 0
        image_index = 0
        video_index = 0
        prompt_index = 0
        current_img_count = 0
        current_vid_count = 0

        # Mutable references for shared callbacks
        total_gen_ref = [0]
        total_chunks_ref = [0]
        total_ttft_ref = [0.0]
        media_count_ref = [0]

        # Shared state for time-sender thread
        from threading import Lock

        _state_lock = Lock()
        _state: Dict[str, Any] = {
            "total_gen": 0,
            "chunk_count": 0,
        }

        _time_thread = make_time_sender(q, worker_id, start_time, duration, _state, _state_lock)

        # Определяем callback media name
        if mixed_mode:
            def _get_media_name() -> str:
                iname = Path(image_paths[image_index % len(image_paths)]).stem if image_paths else ""
                vname = Path(video_paths[video_index % len(video_paths)]).stem if video_paths else ""
                return f"{iname}/{vname}"
        elif has_images:
            _get_media_name = lambda: Path(image_paths[image_index % len(image_paths)]).stem  # noqa: E731
        else:
            _get_media_name = lambda: Path(video_paths[video_index % len(video_paths)]).stem  # noqa: E731

        _on_chunk = make_on_chunk_callback(
            q, worker_id, total_gen_ref, total_chunks_ref, total_ttft_ref,
            _get_media_name,
            media_count_ref=media_count_ref,
        )

        start_event.wait()

        while duration is None or (time.time() - start_time < duration):
            wall_total = time.time() - start_time
            
            selected_images: List[str] = []
            selected_videos: List[str] = []

            if mixed_mode:
                # Изображения: пилообразный паттерн
                if has_images:
                    current_img_count = sawtooth_image_count(call_count, max_images)
                    current_img_count = min(current_img_count, len(image_paths))
                    selected_images = [
                        image_paths[(image_index + i) % len(image_paths)]
                        for i in range(current_img_count)
                    ]
                else:
                    current_img_count = 0

                # Видео: пилообразный паттерн
                if has_videos:
                    current_vid_count = sawtooth_image_count(call_count, max_videos)
                    current_vid_count = min(current_vid_count, len(video_paths))
                    selected_videos = [
                        video_paths[(video_index + i) % len(video_paths)]
                        for i in range(current_vid_count)
                    ]
                else:
                    current_vid_count = 0

                if not selected_images and not selected_videos:
                    break

                prompt = prompts[prompt_index % len(prompts)]
                messages = build_mixed_vision_message(selected_images, selected_videos, prompt)

            elif has_videos:
                # video-only mode
                current_img_count = 0
                current_vid_count = sawtooth_image_count(call_count, max_videos)
                current_vid_count = min(current_vid_count, len(video_paths))
                selected_videos = [
                    video_paths[(video_index + i) % len(video_paths)]
                    for i in range(current_vid_count)
                ]
                if not selected_videos:
                    break
                prompt = prompts[prompt_index % len(prompts)]
                messages = build_video_message(selected_videos, prompt)
                current_vid_count = current_vid_count  # already set

            else:
                # image-only mode
                current_vid_count = 0
                current_img_count = sawtooth_image_count(call_count, max_images)
                current_img_count = min(current_img_count, len(image_paths))
                selected_images = [
                    image_paths[(image_index + i) % len(image_paths)]
                    for i in range(current_img_count)
                ]
                if not selected_images:
                    break
                prompt = prompts[prompt_index % len(prompts)]
                messages = build_vision_message(selected_images, prompt)

            assistant_content = ""
            metrics = None

            try:
                # Обновляем media_count перед вызовом
                media_count_ref[0] = current_img_count + current_vid_count
                session.on_chunk = _on_chunk
                session.on_chunk_args = {
                    "start_time": start_time,
                }

                metrics = session.run(
                    messages=messages,
                    model=MODEL,
                )
                assistant_content = metrics.assistant_content

            except Exception as exc:  # noqa: BLE001
                error_msg = f"[red]Worker {worker_id} stopped: {exc}[/]"
                if not skip_errors:
                    q.put({
                        "type": "error_stop",
                        "id": worker_id,
                        "error": str(exc),
                        "calls": call_count,
                        "wall": format_time(time.time() - start_time),
                    })
                    break
                assistant_content = error_msg

            call_count += 1
            wall_total = time.time() - start_time

            # Имя первого медиа каждого типа
            img_name = Path(selected_images[0]).stem if selected_images else ""
            vid_name = Path(selected_videos[0]).stem if selected_videos else ""

            if metrics is None:
                send_mixed_stats(q, worker_id, call_count, total_gen,
                                 wall_total=wall_total,
                                 img_name=img_name, vid_name=vid_name,
                                 img_count=current_img_count, vid_count=current_vid_count,
                                 tail=assistant_content[:80] if assistant_content else "error",
                                 error=True)
                if mixed_mode:
                    image_index += current_img_count
                    video_index += current_vid_count
                elif has_videos:
                    video_index += current_vid_count
                else:
                    image_index += current_img_count
                prompt_index += 1
                continue

            completion_tokens = metrics.completion_tokens
            chunk_count = metrics.chunk_count

            if completion_tokens == 0:
                send_mixed_stats(q, worker_id, call_count, total_gen,
                                 wall_total=wall_total,
                                 img_name=img_name, vid_name=vid_name,
                                 img_count=current_img_count, vid_count=current_vid_count,
                                 tail="empty", error=True)
                if mixed_mode:
                    image_index += current_img_count
                    video_index += current_vid_count
                elif has_videos:
                    video_index += current_vid_count
                else:
                    image_index += current_img_count
                prompt_index += 1
                continue

            if metrics.ttft is not None:
                turn_ttft = metrics.ttft
                total_ttft += turn_ttft
                ttft_count += 1
            else:
                turn_ttft = 0

            total_gen += completion_tokens
            total_chunks += chunk_count

            total_gen_ref[0] = total_gen
            total_chunks_ref[0] = total_chunks
            total_ttft_ref[0] = total_ttft

            with _state_lock:
                _state["total_gen"] = total_gen
                _state["chunk_count"] = 0

            send_mixed_stats(q, worker_id, call_count, metrics,
                                 total_gen=total_gen,
                                 total_chunks=total_chunks,
                                 total_ttft=total_ttft,
                                 wall_total=wall_total,
                                 img_name=img_name, vid_name=vid_name,
                                 img_count=current_img_count, vid_count=current_vid_count,
                                 tail=assistant_content[-80:] if assistant_content else "")

            if mixed_mode:
                image_index += current_img_count
                video_index += current_vid_count
            elif has_videos:
                video_index += current_vid_count
            else:
                image_index += current_img_count
            prompt_index += 1

    except Exception:
        q.put({
            "type": "error",
            "id": worker_id,
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Mixed stats sender (supports image-only, video-only, and mixed)
# ---------------------------------------------------------------------------

def send_mixed_stats(
    q: Any,
    worker_id: int,
    call_count: int,
    metrics_or_total_gen: Any,
    total_gen: int | None = None,
    total_chunks: int | None = None,
    total_ttft: float | None = None,
    wall_total: float | None = None,
    img_name: str = "",
    vid_name: str = "",
    img_count: int = 0,
    vid_count: int = 0,
    tail: str = "",
    *,
    error: bool = False,
) -> None:
    """Отправляет статистику, поддерживая image-only / video-only / mixed.

    Может использоваться как для успешных вызовов (с StreamMetrics),
    так и для ошибок/пустых ответов (error=True).
    """
    if error:
        msg: dict[str, Any] = {
            "type": "stats",
            "id": worker_id,
            "calls": call_count,
            "g": metrics_or_total_gen if isinstance(metrics_or_total_gen, int) else 0,
            "cg": 0,
            "speed": 0,
            "avg_speed": 0,
            "inst_speed": 0,
            "ttft": 0,
            "tail": tail,
            "wall": format_time(wall_total if wall_total else 0),
        }
        if img_count > 0 or vid_count > 0:
            msg["media_img"] = img_name
            msg["media_img_count"] = img_count
            msg["media_vid"] = vid_name
            msg["media_vid_count"] = vid_count
        # For compatibility with old LiveTable update_stats
        msg["media"] = img_name or vid_name
        msg["media_count"] = img_count or vid_count
        q.put(msg)
        return

    if isinstance(metrics_or_total_gen, int):
        # Это вызов send_error_stats вместо send_stats
        avg_speed = metrics_or_total_gen / wall_total if wall_total and wall_total > 0 else 0
        msg = {
            "type": "stats",
            "id": worker_id,
            "calls": call_count,
            "g": metrics_or_total_gen,
            "cg": 0,
            "speed": 0,
            "avg_speed": round(avg_speed, 1),
            "inst_speed": 0,
            "ttft": 0,
            "tail": tail,
            "wall": format_time(wall_total) if wall_total else "",
        }
    else:
        metrics = metrics_or_total_gen
        msg = {
            "type": "stats",
            "id": worker_id,
            "calls": call_count,
            "g": total_gen,
            "cg": metrics.completion_tokens,
            "chunks": total_chunks,
            "est_gen": total_gen,
            "speed": round(metrics.call_speed, 1),
            "avg_speed": round(total_gen / wall_total, 1) if wall_total and wall_total > 0 else 0,
            "inst_speed": round(metrics.instant_speed, 1),
            "ttft": round(metrics.ttft, 2) if metrics.ttft is not None else 0,
            "ttft_sum": round(total_ttft, 2) if total_ttft is not None else 0,
            "tail": tail,
            "wall": format_time(wall_total) if wall_total else "",
        }

    if img_count > 0 or vid_count > 0:
        msg["media_img"] = img_name
        msg["media_img_count"] = img_count
        msg["media_vid"] = vid_name
        msg["media_vid_count"] = vid_count
    # For compatibility with old VisionLiveTable update_stats
    msg["media"] = img_name or vid_name
    msg["media_count"] = img_count or vid_count

    q.put(msg)


# ---------------------------------------------------------------------------
# Live Table
# ---------------------------------------------------------------------------

class VisionLiveTable(BaseLiveTable):
    """Rich Live table for vision benchmark with dynamic media columns.

    Three layouts:
      - image-only: "Imgs" / "Image" columns (single media)
      - video-only: "Vid" / "Video" columns (single media)
      - mixed:     "Imgs"+"Vid" and "Image"+"Video" (dual media)
    """

    def __init__(
        self,
        duration: Optional[int],
        total_workers: int,
        response_width: int = 60,
        max_images: int = 0,
        max_videos: int = 0,
    ) -> None:
        super().__init__(duration, total_workers, response_width)
        self.mixed_mode = max_images > 0 and max_videos > 0
        self.video_only = max_images == 0 and max_videos > 0

    def mark_started(self, worker_id: int, **kwargs: Any) -> None:
        w: dict[str, Any] = {
            "calls": 0,
            "gen": 0,
            "gen_est": 0,
            "chunks": 0,
            "call_gen": 0,
            "speed": 0,
            "avg": 0,
            "ttft": 0,
            "ttft_sum": 0,
            "wall": "",
            "tail": "[dim]waiting...[/]",
            "media": "",
            "media_count": 0,
        }
        if self.mixed_mode:
            w.update({
                "media_img": "",
                "media_vid": "",
                "media_img_count": 0,
                "media_vid_count": 0,
            })
        self.workers[worker_id] = w

    def update_stats(self, msg: Dict[str, Any]) -> None:
        super().update_stats(msg)
        w = self.workers[msg["id"]]
        self._merge(w, {
            "media": msg.get("media") or msg.get("img", ""),
            "media_count": msg.get("media_count", msg.get("img_count", 0)),
        })
        if self.mixed_mode:
            self._merge(w, {
                "media_img": msg.get("media_img", ""),
                "media_vid": msg.get("media_vid", ""),
                "media_img_count": msg.get("media_img_count", 0),
                "media_vid_count": msg.get("media_vid_count", 0),
            })

    def update_live(self, msg: Dict[str, Any]) -> None:
        super().update_live(msg)
        w = self.workers[msg["id"]]
        self._merge(w, {
            "media": msg.get("media", ""),
        })
        if "media_count" in msg:
            w["media_count"] = msg["media_count"]
        if self.mixed_mode:
            if "media_img" in msg:
                w["media_img"] = msg["media_img"]
            if "media_vid" in msg:
                w["media_vid"] = msg["media_vid"]
            if "media_img_count" in msg:
                w["media_img_count"] = msg["media_img_count"]
            if "media_vid_count" in msg:
                w["media_vid_count"] = msg["media_vid_count"]

    def render(self) -> Table:
        if self.mixed_mode:
            table = self._make_dual_media_table()
            for wid in sorted(self.workers.keys()):
                self._render_worker_row(table, wid, self.workers[wid], dual_media=True)
        elif self.video_only:
            table = self._make_media_table("Vid", "Video")
            for wid in sorted(self.workers.keys()):
                self._render_worker_row(table, wid, self.workers[wid], media=True)
        else:
            table = self._make_media_table("Imgs", "Image")
            for wid in sorted(self.workers.keys()):
                self._render_worker_row(table, wid, self.workers[wid], media=True)

        return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    workers: int = 4,
    duration: Optional[int] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    images_dir: Optional[str] = None,
    videos_dir: Optional[str] = None,
    max_images: int = 1,
    max_videos: int = 0,
    skip_errors: bool = False,
    response_width: int = 60,
    prompts: Optional[List[str]] = None,
) -> None:
    """Запускает vision-бенчмарк (изображения, видео или смешанный режим).

    Args:
        workers: Количество параллельных воркеров.
        duration: Длительность в секундах (None = без ограничения).
        base_url: Переопределение BASE_URL.
        api_key: Переопределение API_KEY.
        model: Переопределение MODEL.
        images_dir: Директория с изображениями.
        videos_dir: Директория с видео.
        max_images: Максимум изображений в одном запросе.
        max_videos: Максимум видео в одном запросе.
        skip_errors: Если True — продолжать после ошибки.
        response_width: Ширина колонки Response.
        prompts: Кастомные промпты.
    """
    apply_config(base_url=base_url, api_key=api_key, model=model)

    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    mixed_mode = max_images > 0 and max_videos > 0
    video_only = max_videos > 0 and max_images == 0

    if prompts is None:
        if video_only:
            prompts = list(DEFAULT_VIDEO_PROMPTS)
        else:
            prompts = list(DEFAULT_PROMPTS)

    # --- Подготовка медиа ---
    image_paths: List[str] = []
    video_paths: List[str] = []

    # --- Изображения ---
    if max_images > 0:
        temp_img_dir = Path(os.path.expanduser("~/.llm-speed-benchmark/tmp/vision_images"))

        if images_dir is not None:
            found = discover_images(images_dir)
            image_paths = [str(p) for p in found]
            if not image_paths:
                print(f"Ошибка: изображения не найдены в {images_dir}")
                sys.exit(1)
            print(f"Найдено {len(image_paths)} изображений в {images_dir}")
        else:
            found = discover_images(temp_img_dir)
            if found:
                image_paths = [str(p) for p in found]
                print(f"Найдено {len(image_paths)} изображений в {temp_img_dir}")
            else:
                gen_count = max(len(prompts) * 2, max_images)
                print(f"Изображения не найдены, генерирую {gen_count} тестовых...")
                generated = generate_test_images(temp_img_dir, count=gen_count)
                image_paths = [str(p) for p in generated]
                print(f"  Создано: {temp_img_dir}")

    # --- Видео ---
    if max_videos > 0:
        temp_vid_dir = Path(os.path.expanduser("~/.llm-speed-benchmark/tmp/vision_videos"))

        if videos_dir is not None:
            found = discover_videos(videos_dir)
            video_paths = [str(p) for p in found]
            if not video_paths:
                print(f"Ошибка: видео не найдены в {videos_dir}")
                sys.exit(1)
            print(f"Найдено {len(video_paths)} видео в {videos_dir}")
        else:
            found = discover_videos(temp_vid_dir)
            if found:
                video_paths = [str(p) for p in found]
                print(f"Найдено {len(video_paths)} видео в {temp_vid_dir}")
            else:
                gen_count = max(len(prompts) * 2, max_videos)
                print(f"Видео не найдены, генерирую {gen_count} тестовых...")
                try:
                    generated = generate_test_videos(temp_vid_dir, count=gen_count)
                    video_paths = [str(p) for p in generated]
                    print(f"  Создано: {temp_vid_dir}")
                except ValueError as exc:
                    print(f"Ошибка: {exc}")
                    sys.exit(1)

    if not image_paths and not video_paths:
        print("Ошибка: нет медиа для бенчмарка.")
        sys.exit(1)

    # --- Заголовок ---
    mode_label = "Mixed" if mixed_mode else ("Video" if video_only else "Vision")
    print("=" * 70)
    print(f"  LLM {mode_label} Benchmark")
    print("=" * 70)
    print(f"  Модель:         {_u.MODEL}")
    print(f"  Воркеров:       {workers}")
    if max_videos > 0:
        print(f"  Видео:          {len(video_paths)}")
        print(f"  Max vids/req:   {max_videos}")
    if max_images > 0:
        print(f"  Изображений:    {len(image_paths)}")
        print(f"  Max imgs/req:   {max_images}")
    print(f"  Промптов:       {len(prompts)}")
    if duration is not None:
        print(f"  Длительность:   {duration}с")
    print("=" * 70)
    print()

    # --- Запуск воркеров ---
    q: "Queue" = Queue()
    start_event = Event()
    processes: List[Process] = []

    for i in range(workers):
        p = Process(
            target=_worker,
            args=(i, q, start_event, image_paths, video_paths, duration, prompts,
                  max_images, max_videos, skip_errors),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Live table ---
    table = VisionLiveTable(duration, workers, response_width,
                            max_images=max_images, max_videos=max_videos)
    live = Live(table.render(), console=table.console, refresh_per_second=4)
    live.start()

    stop_requested = False

    def _handle_signal(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Старт всех воркеров одновременно
    start_event.set()

    try:
        while True:
            if stop_requested:
                break

            # Проверка duration
            if duration is not None:
                alive = any(p.is_alive() for p in processes)
                if not alive:
                    break

            try:
                msg = q.get(timeout=0.5)
            except Exception:
                msg = None

            if msg is None:
                continue

            msg_type = msg.get("type")
            if msg_type == "start":
                table.mark_started(msg["id"])
            elif msg_type == "stats":
                table.update_stats(msg)
            elif msg_type == "live":
                table.update_live(msg)
            elif msg_type == "time":
                table.update_time(msg)
            elif msg_type == "error":
                table.mark_error(msg["id"], msg.get("traceback", ""))
            elif msg_type == "error_stop":
                table.mark_stopped(msg["id"], msg.get("error", "unknown"))

            live.update(table.render())

    except KeyboardInterrupt:
        stop_requested = True

    # --- Остановка ---
    for p in processes:
        p.terminate()
    for p in processes:
        p.join(timeout=3)

    live.stop()

    # --- Итоги ---
    print()
    print("=" * 70)
    print(f"  ИТОГИ ({mode_label} Benchmark)")
    print("=" * 70)

    for wid in sorted(table.workers.keys()):
        w = table.workers[wid]
        err = table._errors.get(wid)
        if err:
            print(f"  Воркер {wid}: ОШИБКА")
            print(f"    {err[:200]}")
            continue
        print(f"  Воркер {wid}:")
        print(f"    Вызовов:    {w.get('calls', 0)}")
        print(f"    Gen total:  {w.get('gen', 0):,} tok")
        print(f"    Avg speed:  {w.get('avg', 0):.1f} tok/s")
        print(f"    TTFT sum:   {w.get('ttft_sum', 0):.1f}s")
        print(f"    Wall time:  {w.get('wall', 'N/A')}")

    # Общее
    total_calls = sum(w.get("calls", 0) for w in table.workers.values())
    total_gen = sum(w.get("gen", 0) for w in table.workers.values())
    print(f"  Всего вызовов: {total_calls}")
    print(f"  Всего токенов: {total_gen:,}")
    print("=" * 70)


def cli() -> None:
    """CLI entry point для bench_vision."""
    parser = argparse.ArgumentParser(
        prog="bench_vision",
        description="Бенчмарк скорости vision-модели (мультимодальная LLM)",
    )
    add_common_args(parser)
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Количество параллельных воркеров (по умолчанию: 4)",
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=None,
        help="Длительность в секундах (без параметра -- без ограничения)",
    )
    parser.add_argument(
        "--images", type=str, default=None,
        help="Директория с изображениями для теста",
    )
    parser.add_argument(
        "--videos", type=str, default=None,
        help="Директория с видео для теста",
    )
    parser.add_argument(
        "--response-width", type=int, default=60,
        help="Ширина колонки Response (по умолчанию: 60)",
    )
    parser.add_argument(
        "--prompt", "-p", type=str, action="append", default=None,
        help="Кастомный промпт (можно указать несколько). Переопределяет дефолтные.",
    )
    parser.add_argument(
        "--max-images", type=int, default=1,
        help=("Максимум изображений в одном запросе (пилообразный паттерн: N..1..N-1). "
              "По умолчанию: 1. Укажите 0 чтобы отключить изображения."),
    )
    parser.add_argument(
        "--max-videos", type=int, default=0,
        help=("Максимум видео в одном запросе (пилообразный паттерн: N..1..N-1). "
              "По умолчанию: 0 (видео отключены). "
              "Если указать >0 вместе с --max-images>0 -- смешанный режим."),
    )
    parser.add_argument(
        "--skip-errors", action="store_true", default=False,
        help="Продолжать после ошибки (по умолчанию воркер останавливается)",
    )

    args = parser.parse_args()

    # Если оба нуля — даём пользователю внятную ошибку
    if args.max_images == 0 and args.max_videos == 0:
        print("Ошибка: хотя бы один из --max-images или --max-videos должен быть > 0.")
        sys.exit(1)

    prompts = args.prompt if args.prompt else None

    run_benchmark(
        workers=args.workers,
        duration=args.duration,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        images_dir=args.images,
        videos_dir=args.videos,
        max_images=args.max_images,
        max_videos=args.max_videos,
        skip_errors=args.skip_errors,
        response_width=args.response_width,
        prompts=prompts,
    )


if __name__ == "__main__":
    cli()