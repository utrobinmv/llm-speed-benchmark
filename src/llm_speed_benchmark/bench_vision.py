#!/usr/bin/env python3
"""
llm_speed_benchmark/bench_vision.py

Бенчмарк скорости vision-модели (мультимодальная LLM) через OpenAI-compatible API.

Отправляет изображения модели с запросом описания, измеряет:
  - TTFT (Time To First Token)
  - Скорость генерации (токенов/сек)
  - Мгновенную скорость
  - Общее время обработки

Поддерживает N параллельных воркеров (multiprocessing) с Rich Live-таблицей.

Использование:
  bench_vision
  bench_vision --workers 4 --duration 120
  bench_vision --images ~/workspace/data/benchmark_images/
  bench_vision --workers 8 --generate 20
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
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.box import DOUBLE

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
    build_vision_message,
    discover_images,
    generate_test_images,
    sawtooth_image_count,
)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(  # type: ignore[reportInvalidTypeForm]
    worker_id: int,
    q: "Queue",  # type: ignore[reportInvalidTypeForm]
    start_event: "Event",  # type: ignore[reportInvalidTypeForm]
    image_paths: List[str],
    duration: Optional[int],
    prompts: List[str],
    max_images: int,
    skip_errors: bool = False,
) -> None:
    """Воркер: загружает изображения и отправляет их модели.

    Каждый воркер циклически проходит по всем изображениям, используя
    разные промпты. Количество изображений в запросе меняется по
    пиломобразному паттерну: max, max-1, ..., 1, 2, ..., max-1.

    При ошибке воркер останавливается (skip_errors=False) или продолжает
    со следующим запросом (skip_errors=True).

    Args:
        worker_id: ID воркера.
        q: Queue для сообщений.
        start_event: Event для синхронизации старта.
        image_paths: Пути к изображениям.
        duration: Длительность в секундах (None = без ограничения).
        prompts: Список промптов для ротации.
        max_images: Максимум изображений в одном запросе.
        skip_errors: Если True — продолжать после ошибки.
    """
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)
        session = StreamSession(client)

        q.put({"type": "start", "id": worker_id, "images": len(image_paths)})

        start_time = time.time()
        call_count = 0
        total_gen = 0
        total_chunks = 0
        total_ttft = 0.0
        ttft_count = 0
        img_index = 0
        prompt_index = 0
        current_img_count = max_images  # для live callback

        # Shared state для потока time_sender
        from threading import Lock, Thread

        _state_lock = Lock()
        _state: Dict[str, Any] = {
            "total_gen": 0,
            "tokens_per_chunk": 1.0,
            "chunk_count": 0,
        }

        def _time_sender() -> None:
            while True:
                time.sleep(1.0)
                wall = time.time() - start_time
                if wall > 0:
                    with _state_lock:
                        tg = _state["total_gen"]
                        tpc = _state["tokens_per_chunk"]
                        cc = _state["chunk_count"]
                    est_gen = tg + (cc * tpc)
                    avg_sp = round(est_gen / wall, 1) if wall > 0 else 0
                    q.put({
                        "type": "time",
                        "id": worker_id,
                        "wall": format_time(wall),
                        "avg": avg_sp,
                    })

        _time_thread = Thread(target=_time_sender, daemon=True)
        _time_thread.start()

        # Callback для live-обновлений
        def _on_chunk(
            chunk_count: int,
            inst_sp: float,
            avg_sp: float,
            ttft: Optional[float],
            tail: str,
            wall_elapsed: float,
        ) -> None:
            current_ttft = ttft if ttft is not None else 0
            ttft_sum_live = total_ttft + current_ttft
            q.put({
                "type": "live",
                "id": worker_id,
                "tail": tail,
                "tok": chunk_count,
                "est_tok": round(total_gen + (chunk_count * session.tokens_per_chunk)),
                "chunks": total_chunks + chunk_count,
                "inst": inst_sp,
                "avg": avg_sp,
                "ttft": ttft if ttft is not None else 0,
                "ttft_sum": round(ttft_sum_live, 2),
                "wall": format_time(wall_elapsed),
                "img": Path(image_paths[img_index % len(image_paths)]).stem,
                "img_count": current_img_count,
            })

        start_event.wait()

        while duration is None or (time.time() - start_time < duration):
            if not image_paths:
                break

            wall_total = time.time() - start_time

            # Пилообразный паттерн: max, max-1, ..., 1, 2, ..., max-1
            num_images = sawtooth_image_count(call_count, max_images)
            # Ограничиваем количеством доступных изображений
            num_images = min(num_images, len(image_paths))
            current_img_count = num_images

            # Выбираем изображения и промпт
            selected_images: List[str] = [
                image_paths[(img_index + i) % len(image_paths)]
                for i in range(num_images)
            ]
            prompt = prompts[prompt_index % len(prompts)]

            # Строим vision message
            messages = build_vision_message(selected_images, prompt)

            assistant_content = ""
            metrics = None

            try:
                session.on_chunk = _on_chunk
                session.on_chunk_args = {
                    "total_gen": total_gen,
                    "start_time": start_time,
                }
                session.call_count = call_count

                metrics = session.run(
                    messages=messages,
                    model=MODEL,
                )
                assistant_content = metrics.assistant_content

            except Exception as exc:  # noqa: BLE001
                error_msg = f"[red]Worker {worker_id} stopped: {exc}[/]"
                q.put({
                    "type": "error_stop",
                    "id": worker_id,
                    "error": str(exc),
                    "calls": call_count,
                    "wall": format_time(time.time() - start_time),
                })
                if not skip_errors:
                    # Воркер останавливается
                    break
                assistant_content = error_msg

            call_count += 1
            wall_total = time.time() - start_time

            # Если ошибка в stream
            if metrics is None:
                q.put({
                    "type": "stats",
                    "id": worker_id,
                    "calls": call_count,
                    "img": Path(selected_images[0]).stem,
                    "img_count": current_img_count,
                    "g": total_gen,
                    "cg": 0,
                    "speed": 0,
                    "avg_speed": 0,
                    "inst_speed": 0,
                    "ttft": 0,
                    "tail": assistant_content[:80] if assistant_content else "error",
                    "wall": format_time(wall_total),
                })
                img_index += num_images
                prompt_index += 1
                continue

            completion_tokens = metrics.completion_tokens
            chunk_count = metrics.chunk_count

            # Пустой ответ
            if completion_tokens == 0:
                q.put({
                    "type": "stats",
                    "id": worker_id,
                    "calls": call_count,
                    "img": Path(selected_images[0]).stem,
                    "img_count": current_img_count,
                    "g": total_gen,
                    "cg": 0,
                    "speed": 0,
                    "avg_speed": 0,
                    "inst_speed": 0,
                    "ttft": 0,
                    "tail": "empty",
                    "wall": format_time(wall_total),
                })
                img_index += num_images
                prompt_index += 1
                continue

            # TTFT
            if metrics.ttft is not None:
                turn_ttft = metrics.ttft
                total_ttft += turn_ttft
                ttft_count += 1
            else:
                turn_ttft = 0

            total_gen += completion_tokens
            total_chunks += chunk_count

            # Обновляем shared state
            with _state_lock:
                _state["total_gen"] = total_gen
                _state["tokens_per_chunk"] = session.tokens_per_chunk
                _state["chunk_count"] = 0

            avg_speed = total_gen / wall_total if wall_total > 0 else 0

            # Финальная статистика
            q.put({
                "type": "stats",
                "id": worker_id,
                "calls": call_count,
                "img": Path(selected_images[0]).stem,
                "img_count": current_img_count,
                "g": total_gen,
                "cg": completion_tokens,
                "chunks": total_chunks,
                "est_gen": total_gen,
                "speed": round(metrics.call_speed, 1),
                "avg_speed": round(avg_speed, 1),
                "inst_speed": round(metrics.instant_speed, 1),
                "ttft": round(turn_ttft, 2),
                "ttft_sum": round(total_ttft, 2),
                "tail": assistant_content[-80:] if assistant_content else "",
                "wall": format_time(wall_total),
            })

            img_index += num_images
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

class VisionLiveTable:
    """Rich Live таблица для vision-бенчмарка."""

    def __init__(self, duration: Optional[int], total_workers: int, response_width: int = 60) -> None:
        self.duration = duration
        self.total_workers = total_workers
        self.response_width = response_width
        self.workers: Dict[int, Dict[str, Any]] = {}
        self._errors: Dict[int, str] = {}
        self.console = Console()

    @staticmethod
    def _merge(w: Dict[str, Any], data: Dict[str, Any]) -> None:
        for k, v in data.items():
            if v is not None:
                w[k] = v

    def mark_started(self, worker_id: int, image_count: int = 0) -> None:
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
            "img": "",
            "img_count": 0,
            "images": image_count,
        }

    def update_stats(self, msg: Dict[str, Any]) -> None:
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "calls": msg.get("calls"),
            "img": msg.get("img"),
            "img_count": msg.get("img_count"),
            "g": msg.get("g"),
            "gen_est": msg.get("est_gen"),
            "chunks": msg.get("chunks"),
            "call_gen": msg.get("cg"),
            "speed": msg.get("inst_speed"),
            "avg": msg.get("avg_speed"),
            "ttft": msg.get("ttft"),
            "ttft_sum": msg.get("ttft_sum"),
            "wall": msg.get("wall"),
            "tail": msg.get("tail"),
        })

    def update_live(self, msg: Dict[str, Any]) -> None:
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "speed": msg.get("inst"),
            "avg": msg.get("avg"),
            "ttft": msg.get("ttft"),
            "ttft_sum": msg.get("ttft_sum"),
            "gen_est": msg.get("est_tok"),
            "chunks": msg.get("chunks"),
            "wall": msg.get("wall"),
            "tail": msg.get("tail"),
            "img": msg.get("img", ""),
            "img_count": msg.get("img_count", 0),
        })

    def update_time(self, msg: Dict[str, Any]) -> None:
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "wall": msg.get("wall"),
            "avg": msg.get("avg"),
        })

    def mark_error(self, worker_id: int, traceback_str: str) -> None:
        self._errors[worker_id] = traceback_str

    def mark_stopped(self, worker_id: int, error_msg: str) -> None:
        """Воркер остановлен из-за ошибки."""
        w = self.workers.setdefault(worker_id, {})
        w["stopped"] = True
        w["error"] = error_msg

    @staticmethod
    def _clean_tail(tail: str, max_len: int = 60) -> str:
        """Очищает текст для отображения в таблице."""
        if not tail:
            return tail
        # Убираем wide символы и новые строки
        clean = ""
        in_tag = False
        for c in tail:
            if c == "[":
                in_tag = True
                clean += c
                continue
            if c == "]" and in_tag:
                in_tag = False
                clean += c
                continue
            if in_tag:
                clean += c
                continue
            cp = ord(c)
            if cp >= 0x4E00 and cp <= 0x9FFF:
                clean += "."
            elif c in "\n\r\t":
                clean += " "
            elif c.isprintable():
                clean += c
            else:
                clean += "."
        # Схлопываем пробелы
        prev = None
        while prev != clean:
            prev = clean
            parts = []
            in_tag = False
            prev_space = False
            for c in prev:
                if c == "[":
                    in_tag = True
                    parts.append(c)
                    prev_space = False
                    continue
                if c == "]" and in_tag:
                    in_tag = False
                    parts.append(c)
                    prev_space = False
                    continue
                if in_tag:
                    parts.append(c)
                    prev_space = False
                    continue
                if c == " " and prev_space:
                    continue
                prev_space = c == " "
                parts.append(c)
            clean = "".join(parts)
        clean = clean.strip()
        if len(clean) > max_len:
            clean = clean[:max_len - 3] + "..."
        return clean

    def render(self) -> Table:
        table = Table(box=DOUBLE, show_header=True)

        table.add_column("W#", style="cyan", width=4, justify="right")
        table.add_column("Calls", style="magenta", width=6, justify="right")
        table.add_column("Imgs", style="bold yellow", width=5, justify="right")
        table.add_column("Image", style="green", width=14)
        table.add_column("Gen", style="yellow", width=8, justify="right")
        table.add_column("Call", style="white", width=7, justify="right")
        table.add_column("Speed", style="bold green", width=9, justify="right")
        table.add_column("Avg", style="bold blue", width=8, justify="right")
        table.add_column("TTFT", style="red", width=8, justify="right")
        table.add_column("TTFT sum", style="red", width=10, justify="right")
        table.add_column("Wall", style="dim", width=7, justify="right")
        table.add_column("Response", style="dim white", width=self.response_width)

        for wid in sorted(self.workers.keys()):
            w = self.workers[wid]
            err = self._errors.get(wid)

            if err:
                table.add_row(
                    str(wid),
                    str(w.get("calls", 0)),
                    "",
                    "",
                    "ERROR",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "[red]SEE LOG[/]",
                )
                continue

            if w.get("stopped"):
                error_text = self._clean_tail(w.get("error", "unknown"), self.response_width)
                table.add_row(
                    str(wid),
                    str(w.get("calls", 0)),
                    "",
                    "",
                    "STOPPED",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"[red]{error_text}[/]",
                )
                continue

            gen_val = w.get("gen_est", w.get("gen", 0))
            tail = self._clean_tail(w.get("tail", ""), self.response_width)
            img_name = w.get("img", "")
            img_count = w.get("img_count", 0)

            table.add_row(
                str(wid),
                str(w.get("calls", 0)),
                str(img_count),
                img_name,
                f"{gen_val:,}",
                f"{w.get('call_gen', 0):,}",
                f"{w.get('speed', 0):.1f} t/s",
                f"{w.get('avg', 0):.1f} t/s",
                f"{w.get('ttft', 0):.2f}s",
                f"{w.get('ttft_sum', 0):.1f}s",
                w.get("wall", ""),
                tail,
            )

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
    generate_images: Optional[int] = None,
    max_images: int = 1,
    skip_errors: bool = False,
    response_width: int = 60,
    prompts: Optional[List[str]] = None,
) -> None:
    """Запускает vision-бенчмарк.

    Args:
        workers: Количество параллельных воркеров.
        duration: Длительность в секундах (None = без ограничения).
        base_url: Переопределение BASE_URL.
        api_key: Переопределение API_KEY.
        model: Переопределение MODEL.
        images_dir: Директория с изображениями.
        generate_images: Сгенерировать N тестовых изображений.
        max_images: Максимум изображений в одном запросе.
        skip_errors: Если True — продолжать после ошибки (по умолчанию воркер останавливается).
        response_width: Ширина колонки Response.
        prompts: Кастомные промпты (по умолчанию DEFAULT_PROMPTS).
    """
    apply_config(base_url=base_url, api_key=api_key, model=model)

    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    if prompts is None:
        prompts = list(DEFAULT_PROMPTS)

    # --- Подготовка изображений ---
    temp_dir = Path(os.path.expanduser("~/.llm-speed-benchmark/tmp/vision_images"))
    image_paths: List[str] = []

    if generate_images is not None and generate_images > 0:
        print(f"Генерация {generate_images} тестовых изображений...")
        generated = generate_test_images(temp_dir, count=generate_images)
        image_paths = [str(p) for p in generated]
        print(f"  Создано: {temp_dir}")
    elif images_dir is not None:
        found = discover_images(images_dir)
        image_paths = [str(p) for p in found]
        if not image_paths:
            print(f"Ошибка: изображения не найдены в {images_dir}")
            sys.exit(1)
        print(f"Найдено {len(image_paths)} изображений в {images_dir}")
    else:
        # По умолчанию ищем в ~/.llm-speed-benchmark/tmp/vision_images/
        found = discover_images(temp_dir)
        if found:
            image_paths = [str(p) for p in found]
            print(f"Найдено {len(image_paths)} изображений в {temp_dir}")
        else:
            # Генерируем по умолчанию
            print(f"Изображения не найдены, генерирую {len(DEFAULT_PROMPTS) * 2} тестовых...")
            generated = generate_test_images(temp_dir, count=len(DEFAULT_PROMPTS) * 2)
            image_paths = [str(p) for p in generated]
            print(f"  Создано: {temp_dir}")

    if not image_paths:
        print("Ошибка: нет изображений для бенчмарка.")
        sys.exit(1)

    # --- Заголовок ---
    print("=" * 70)
    print("  LLM Vision Benchmark")
    print("=" * 70)
    print(f"  Модель:         {_u.MODEL}")
    print(f"  Воркеров:       {workers}")
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
            args=(i, q, start_event, image_paths, duration, prompts, max_images, skip_errors),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Live table ---
    table = VisionLiveTable(duration, workers, response_width)
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
                table.mark_started(msg["id"], msg.get("images", 0))
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
        "--generate", type=int, default=None,
        help="Сгенерировать N тестовых изображений (вместо загрузки из директории)",
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
        "--skip-errors", action="store_true", default=False,
        help="Продолжать после ошибки (по умолчанию воркер останавливается)",
    )

    args = parser.parse_args()

    prompts = args.prompt if args.prompt else None

    run_benchmark(
        workers=args.workers,
        duration=args.duration,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        images_dir=args.images,
        generate_images=args.generate,
        max_images=args.max_images,
        skip_errors=args.skip_errors,
        response_width=args.response_width,
        prompts=prompts,
    )


if __name__ == "__main__":
    cli()
