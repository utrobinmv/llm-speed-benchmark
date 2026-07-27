#!/usr/bin/env python3
"""
llm_speed_benchmark/bench_vision.py

Бенчмарк скорости vision/video-модели (мультимодальная LLM) через OpenAI-compatible API.

Отправляет изображения или видео модели с запросом описания, измеряет:
  - TTFT (Time To First Token)
  - Скорость генерации (токенов/сек)
  - Мгновенную скорость
  - Общее время обработки

Поддерживает N параллельных воркеров (multiprocessing) с Rich Live-таблицей.

Использование:
  bench_vision
  bench_vision --workers 4 --duration 120
  bench_vision --images ~/workspace/data/benchmark_images/
  bench_vision --max-images 4
  bench_vision --videos ~/workspace/data/test_videos/
  bench_vision --max-videos 4
  bench_vision --max-videos 4 -w 8 -d 60
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


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(  # type: ignore[reportInvalidTypeForm, valid-type]
    worker_id: int,
    q: "Queue",  # type: ignore[reportInvalidTypeForm, valid-type]
    start_event: "Event",  # type: ignore[reportInvalidTypeForm, valid-type]
    media_paths: List[str],
    duration: Optional[int],
    prompts: List[str],
    max_images: int,
    skip_errors: bool = False,
    video_mode: bool = False,
    max_videos: int = 0,
) -> None:
    """Worker: loads media (images or videos) and sends to the model.

    In image mode -- cycles through all images using different prompts.
    The number of images per request follows a sawtooth pattern:
    max, max-1, ..., 1, 2, ..., max-1.

    In video mode -- same pattern with videos.

    Stops on error (skip_errors=False) or continues (skip_errors=True).
    """
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)
        session = StreamSession(client)

        q.put({"type": "start", "id": worker_id, "media": len(media_paths)})

        start_time = time.time()
        call_count = 0
        total_gen = 0
        total_chunks = 0
        total_ttft = 0.0
        ttft_count = 0
        media_index = 0
        prompt_index = 0
        current_count = 1

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

        _on_chunk = make_on_chunk_callback(
            q, worker_id, total_gen_ref, total_chunks_ref, total_ttft_ref,
            lambda: Path(media_paths[media_index % len(media_paths)]).stem,
            media_count_ref=media_count_ref,
        )

        start_event.wait()

        while duration is None or (time.time() - start_time < duration):
            if not media_paths:
                break

            wall_total = time.time() - start_time

            if video_mode:
                num_media = sawtooth_image_count(call_count, max_videos)
                num_media = min(num_media, len(media_paths))
                current_count = num_media
                selected_media = [
                    media_paths[(media_index + i) % len(media_paths)]
                    for i in range(num_media)
                ]
            else:
                num_media = sawtooth_image_count(call_count, max_images)
                num_media = min(num_media, len(media_paths))
                current_count = num_media
                selected_media = [
                    media_paths[(media_index + i) % len(media_paths)]
                    for i in range(num_media)
                ]

            prompt = prompts[prompt_index % len(prompts)]

            if video_mode:
                messages = build_video_message(selected_media, prompt)
            else:
                messages = build_vision_message(selected_media, prompt)

            assistant_content = ""
            metrics = None

            try:
                # Обновляем media_count перед вызовом
                media_count_ref[0] = current_count
                session.on_chunk = _on_chunk
                session.on_chunk_args = {
                    "start_time": start_time,
                }

                metrics = session.run(
                    messages=messages,
                    model=MODEL,
                )
                assistant_content = metrics.assistant_content

            except Exception as exc:  # noqa: BLE0414
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
            media_name = Path(selected_media[0]).stem

            if metrics is None:
                send_error_stats(q, worker_id, call_count, total_gen, wall_total,
                                 media_name, current_count,
                                 assistant_content[:80] if assistant_content else "error")
                media_index += current_count
                prompt_index += 1
                continue

            completion_tokens = metrics.completion_tokens
            chunk_count = metrics.chunk_count

            if completion_tokens == 0:
                send_error_stats(q, worker_id, call_count, total_gen, wall_total,
                                 media_name, current_count, "empty")
                media_index += current_count
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

            send_stats(q, worker_id, call_count, metrics, total_gen, total_chunks,
                       total_ttft, wall_total, media_name, current_count,
                       assistant_content[-80:] if assistant_content else "")

            media_index += current_count
            prompt_index += 1

    except Exception:
        q.put({
            "type": "error",
            "id": worker_id,
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Live Table
# ---------------------------------------------------------------------------

class VisionLiveTable(BaseLiveTable):
    """Rich Live table for vision benchmark (images or video)."""

    def __init__(
        self,
        duration: Optional[int],
        total_workers: int,
        response_width: int = 60,
        video_mode: bool = False,
    ) -> None:
        super().__init__(duration, total_workers, response_width)
        self.video_mode = video_mode

    def mark_started(self, worker_id: int, media_count: int = 0) -> None:
        self.workers[worker_id] = {
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
            "total_media": media_count,
        }

    def update_stats(self, msg: Dict[str, Any]) -> None:
        super().update_stats(msg)
        w = self.workers[msg["id"]]
        self._merge(w, {
            "media": msg.get("media") or msg.get("img", ""),
            "media_count": msg.get("media_count", msg.get("img_count", 0)),
        })

    def update_live(self, msg: Dict[str, Any]) -> None:
        super().update_live(msg)
        w = self.workers[msg["id"]]
        self._merge(w, {
            "media": msg.get("media") or msg.get("img", ""),
        })
        # Only update media_count if explicitly present in the live message
        if "media_count" in msg:
            w["media_count"] = msg["media_count"]

    def render(self) -> Table:
        if self.video_mode:
            table = self._make_media_table("Vid", "Video")
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
    video_mode: bool = False,
) -> None:
    """Запускает vision-бенчмарк (изображения или видео).

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
        video_mode: Если True — режим видео вместо изображений.
    """
    apply_config(base_url=base_url, api_key=api_key, model=model)

    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    if video_mode:
        if prompts is None:
            prompts = list(DEFAULT_VIDEO_PROMPTS)
    else:
        if prompts is None:
            prompts = list(DEFAULT_PROMPTS)

    # --- Подготовка медиа ---
    media_paths: List[str] = []
    if video_mode:
        temp_dir = Path(os.path.expanduser("~/.llm-speed-benchmark/tmp/vision_videos"))

        if videos_dir is not None:
            found = discover_videos(videos_dir)
            media_paths = [str(p) for p in found]
            if not media_paths:
                print(f"Ошибка: видео не найдены в {videos_dir}")
                sys.exit(1)
            print(f"Найдено {len(media_paths)} видео в {videos_dir}")
        else:
            # По умолчанию ищем в ~/.llm-speed-benchmark/tmp/vision_videos/
            found = discover_videos(temp_dir)
            if found:
                media_paths = [str(p) for p in found]
                print(f"Найдено {len(media_paths)} видео в {temp_dir}")
            else:
                # Пробуем сгенерировать (минимум max_videos)
                gen_count = max(len(DEFAULT_VIDEO_PROMPTS) * 2, max_videos)
                print(f"Видео не найдены, генерирую {gen_count} тестовых...")
                try:
                    generated = generate_test_videos(temp_dir, count=gen_count)
                    media_paths = [str(p) for p in generated]
                    print(f"  Создано: {temp_dir}")
                except ValueError as exc:
                    print(f"Ошибка: {exc}")
                    sys.exit(1)
    else:
        temp_dir = Path(os.path.expanduser("~/.llm-speed-benchmark/tmp/vision_images"))

        if images_dir is not None:
            found = discover_images(images_dir)
            media_paths = [str(p) for p in found]
            if not media_paths:
                print(f"Ошибка: изображения не найдены в {images_dir}")
                sys.exit(1)
            print(f"Найдено {len(media_paths)} изображений в {images_dir}")
        else:
            # По умолчанию ищем в ~/.llm-speed-benchmark/tmp/vision_images/
            found = discover_images(temp_dir)
            if found:
                media_paths = [str(p) for p in found]
                print(f"Найдено {len(media_paths)} изображений в {temp_dir}")
            else:
                # Генерируем минимум max_images
                gen_count = max(len(DEFAULT_PROMPTS) * 2, max_images)
                print(f"Изображения не найдены, генерирую {gen_count} тестовых...")
                generated = generate_test_images(temp_dir, count=gen_count)
                media_paths = [str(p) for p in generated]
                print(f"  Создано: {temp_dir}")

    if not media_paths:
        print("Ошибка: нет медиа для бенчмарка.")
        sys.exit(1)

    # --- Заголовок ---
    mode_label = "Video" if video_mode else "Vision"
    print("=" * 70)
    print(f"  LLM {mode_label} Benchmark")
    print("=" * 70)
    print(f"  Модель:         {_u.MODEL}")
    print(f"  Воркеров:       {workers}")
    if video_mode:
        print(f"  Видео:          {len(media_paths)}")
        print(f"  Max vids/req:   {max_videos}")
    else:
        print(f"  Изображений:    {len(media_paths)}")
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
            args=(i, q, start_event, media_paths, duration, prompts, max_images, skip_errors, video_mode, max_videos),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Live table ---
    table = VisionLiveTable(duration, workers, response_width, video_mode=video_mode)
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
                table.mark_started(msg["id"], msg.get("media", 0))
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
    print("  ИТОГИ (Vision Benchmark)")
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
        help="Директория с видео для теста (переключает в видео-режим)",
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
        help="Максимум изображений в одном запросе (пилообразный паттерн: N..1..N-1). По умолчанию: 1",
    )
    parser.add_argument(
        "--max-videos", type=int, default=0,
        help="Максимум видео в одном запросе (пилообразный паттерн: N..1..N-1). По умолчанию: 0",
    )
    parser.add_argument(
        "--skip-errors", action="store_true", default=False,
        help="Продолжать после ошибки (по умолчанию воркер останавливается)",
    )

    args = parser.parse_args()

    # Видео-режим включается если указан --videos или --max-videos > 0
    video_mode = args.videos is not None or args.max_videos > 0

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
        video_mode=video_mode,
    )


if __name__ == "__main__":
    cli()
