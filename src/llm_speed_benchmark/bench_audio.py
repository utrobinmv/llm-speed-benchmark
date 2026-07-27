"""
llm_speed_benchmark/bench_audio.py

Бенчмарк скорости аудио-модели (транскрипция, распознавание речи).
Отправляет аудио файлы через OpenAI-compatible API, измеряет TTFT и скорость генерации.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from rich.live import Live
from rich.table import Table

from .audio_utils import (
    DEFAULT_AUDIO_PROMPTS,
    build_audio_message,
    discover_audio,
    get_audio_paths,
)
from .cli_common import add_common_args, apply_config
from .streaming import StreamSession
from .utils import format_time
from .live_table import BaseLiveTable
from .worker_common import make_time_sender, make_on_chunk_callback, send_stats, send_error_stats


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(  # type: ignore[valid-type]
    worker_id: int,
    q: Queue,  # type: ignore[valid-type]
    start_event: Event,  # type: ignore[valid-type]
    audio_paths: List[str],
    duration: Optional[int],
    prompts: List[str],
    max_audio: int,
    skip_errors: bool,
    response_width: int,
) -> None:
    """Worker for audio benchmark.

    Args:
        worker_id: Worker ID.
        q: Message queue.
        start_event: Start synchronization event.
        audio_paths: Paths to audio files.
        duration: Duration in seconds (None = unlimited).
        prompts: Prompts for requests.
        max_audio: Max audio files per request.
        skip_errors: Continue after error.
        response_width: Response column width.
    """
    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    BASE_URL = _u.BASE_URL
    API_KEY = _u.API_KEY
    MODEL = _u.MODEL

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)
    session = StreamSession(client)

    q.put({
        "type": "start",
        "id": worker_id,
        "media": len(audio_paths),
    })

    start_time = time.time()
    call_count = 0
    total_gen = 0
    total_chunks = 0
    total_ttft = 0.0
    ttft_count = 0
    audio_index = 0
    prompt_index = 0

    # Mutable refs for shared callback
    total_gen_ref: list[int] = [0]
    total_chunks_ref: list[int] = [0]
    total_ttft_ref: list[float] = [0.0]
    media_count_ref: list[int] = [0]

    _time_thread = make_time_sender(q, worker_id, start_time, duration)

    _on_chunk = make_on_chunk_callback(
        q, worker_id, total_gen_ref, total_chunks_ref, total_ttft_ref,
        lambda: Path(audio_paths[audio_index % len(audio_paths)]).stem,
        media_count_ref=media_count_ref,
    )

    start_event.wait()

    try:
        while duration is None or (time.time() - start_time < duration):
            if not audio_paths:
                break

            num_media = min(max_audio, len(audio_paths))
            selected_audio = [
                audio_paths[(audio_index + i) % len(audio_paths)]
                for i in range(num_media)
            ]

            prompt = prompts[prompt_index % len(prompts)]
            messages = build_audio_message(selected_audio, prompt)

            assistant_content = ""
            metrics = None

            try:
                # Обновляем media_count перед вызовом
                media_count_ref[0] = num_media
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
            media_name = Path(selected_audio[0]).stem

            if metrics is None:
                send_error_stats(q, worker_id, call_count, total_gen, wall_total,
                                 media_name, num_media,
                                 assistant_content[:80] if assistant_content else "error")
                audio_index += num_media
                prompt_index += 1
                continue

            completion_tokens = metrics.completion_tokens
            chunk_count = metrics.chunk_count

            if completion_tokens == 0:
                send_error_stats(q, worker_id, call_count, total_gen, wall_total,
                                 media_name, num_media, "empty")
                audio_index += num_media
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

            send_stats(q, worker_id, call_count, metrics, total_gen, total_chunks,
                       total_ttft, wall_total, media_name, num_media,
                       assistant_content[-80:] if assistant_content else "")

            audio_index += num_media
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

class AudioLiveTable(BaseLiveTable):
    """Rich Live table for audio benchmark."""

    def __init__(
        self,
        duration: Optional[int],
        total_workers: int,
        response_width: int = 60,
    ) -> None:
        super().__init__(duration, total_workers, response_width)

    def mark_started(self, worker_id: int, audio_count: int = 0) -> None:
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
            "total_audio": audio_count,
        }

    def update_stats(self, msg: Dict[str, Any]) -> None:
        super().update_stats(msg)
        w = self.workers[msg["id"]]
        self._merge(w, {
            "media": msg.get("media"),
            "media_count": msg.get("media_count"),
        })

    def update_live(self, msg: Dict[str, Any]) -> None:
        super().update_live(msg)
        w = self.workers[msg["id"]]
        self._merge(w, {
            "media": msg.get("media", ""),
        })
        if "media_count" in msg:
            w["media_count"] = msg["media_count"]

    def render(self) -> Table:
        table = self._make_media_table("Audios", "Audio")

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
    audio_dir: Optional[str] = None,
    max_audio: int = 1,
    skip_errors: bool = False,
    response_width: int = 60,
    prompts: Optional[List[str]] = None,
) -> None:
    """Запускает аудио-бенчмарк."""
    apply_config(base_url=base_url, api_key=api_key, model=model)

    if prompts is None:
        prompts = list(DEFAULT_AUDIO_PROMPTS)

    # --- Подготовка аудио ---
    audio_paths: List[str] = []

    if audio_dir is not None:
        found = discover_audio(audio_dir)
        audio_paths = [str(p) for p in found]
        if not audio_paths:
            print(f"Ошибка: аудио не найдено в {audio_dir}")
            sys.exit(1)
        print(f"Найдено {len(audio_paths)} аудио в {audio_dir}")
    else:
        found = get_audio_paths()
        audio_paths = [str(p) for p in found]
        if not audio_paths:
            print("Ошибка: аудиофайлы бандла не найдены")
            sys.exit(1)
        print(f"Бандл: {len(audio_paths)} аудио")

    if not audio_paths:
        print("Ошибка: нет аудио файлов для бенчмарка")
        sys.exit(1)

    # --- Заголовок ---
    print("=" * 70)
    print("  LLM Audio Benchmark")
    print("=" * 70)
    print(f"  Модель:         {model or os.environ.get('MODEL', 'N/A')}")
    print(f"  Воркеров:       {workers}")
    print(f"  Аудио:          {len(audio_paths)}")
    print(f"  Max audio/req:  {max_audio}")
    print(f"  Промптов:       {len(prompts)}")
    if duration:
        print(f"  Длительность:   {duration}с")
    print("=" * 70)
    print()

    # --- Запуск воркеров ---
    q: Queue = Queue()  # type: ignore[valid-type]
    start_event: Event = Event()  # type: ignore[valid-type]
    processes: List[Process] = []

    for i in range(workers):
        p = Process(
            target=_worker,
            args=(i, q, start_event, audio_paths, duration, prompts, max_audio, skip_errors, response_width),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Live table ---
    table = AudioLiveTable(duration, workers, response_width)
    live = Live(table.render(), console=table.console, refresh_per_second=4)
    live.start()

    stop_requested = False

    def _handle_signal(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_event.set()

    try:
        while True:
            if stop_requested:
                break

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
    print("  ИТОГИ (Audio Benchmark)")
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

    total_calls = sum(w.get("calls", 0) for w in table.workers.values())
    total_gen = sum(w.get("gen", 0) for w in table.workers.values())
    print(f"  Всего вызовов: {total_calls}")
    print(f"  Всего токенов: {total_gen:,}")
    print("=" * 70)


def cli() -> None:
    """CLI entry point для bench_audio."""
    parser = argparse.ArgumentParser(
        prog="bench_audio",
        description="Бенчмарк скорости аудио-модели (транскрипция, распознавание речи)",
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
        "--audio", type=str, default=None,
        help="Директория с аудио файлами для теста",
    )
    parser.add_argument(
        "--max-audio", type=int, default=1,
        help="Максимум аудио файлов в одном запросе (по умолчанию: 1)",
    )
    parser.add_argument(
        "--prompt", "-p", action="append", default=None,
        help="Кастомный промпт (можно указать несколько)",
    )
    parser.add_argument(
        "--response-width", type=int, default=60,
        help="Ширина колонки Response в таблице (по умолчанию: 60)",
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
        audio_dir=args.audio,
        max_audio=args.max_audio,
        skip_errors=args.skip_errors,
        response_width=args.response_width,
        prompts=prompts,
    )


if __name__ == "__main__":
    cli()
