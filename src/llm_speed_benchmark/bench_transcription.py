"""
llm_speed_benchmark/bench_transcription.py

Бенчмарк скорости ASR-моделей (Automatic Speech Recognition).
Тестирует модели транскрипции: Whisper, Qwen-ASR и другие через OpenAI-compatible API.

Метрики:
  - RTF (Real-Time Factor): processing_time / audio_duration
  - Speed (tok/s): скорость выдачи токенов транскрипции
  - TTFT: время до первого токена
  - Audio speed: секунд аудио обработано за секунду wall time

Использует endpoint audio.transcriptions (не chat.completions).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.table import Table, box as _rich_box

from .cli_common import add_common_args, apply_config
from .transcription_utils import (
    DEFAULT_TRANSCRIPTION_PROMPTS,
    discover_audio_files,
    filter_by_max_duration,
    generate_audio_at_duration,
    generate_duration_series,
    get_audio_duration,
    get_audio_info,
    load_bundled_audio,
)
from .utils import format_time


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(  # type: ignore[valid-type]
    worker_id: int,
    q: Queue,  # type: ignore[valid-type]
    start_event: Event,  # type: ignore[valid-type]
    audio_tasks: List[Dict[str, Any]],
    duration: Optional[int],
    model_timeout: float,
    prompt: Optional[str],
    language: Optional[str],
    response_format: str,
    skip_errors: bool,
    response_width: int,
    mode: str,
) -> None:
    """Worker для бенчмарка транскрипции ASR.

    Args:
        worker_id: ID воркера.
        q: Очередь сообщений.
        start_event: Событие синхронизации старта.
        audio_tasks: Список задач: [{"path": str, "duration": float}, ...].
        duration: Общее время бенчмарка в секундах (None = без лимита).
        model_timeout: Таймаут запроса к модели.
        prompt: Опциональный промпт.
        language: Опциональный язык.
        response_format: Формат ответа API.
        skip_errors: Продолжать после ошибки.
        response_width: Ширина колонки Response.
        mode: "cycling" или "once".
    """
    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    BASE_URL = _u.BASE_URL
    API_KEY = _u.API_KEY
    MODEL = _u.MODEL

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=model_timeout)

    q.put({
        "type": "start",
        "id": worker_id,
        "tasks": len(audio_tasks),
    })

    start_time = time.time()
    call_count = 0
    total_tokens = 0
    total_audio_seconds = 0.0
    total_processing_time = 0.0
    total_ttft = 0.0
    ttft_count = 0
    task_index = 0

    # Mutable refs для live-обновлений
    total_tokens_ref: list[int] = [0]
    total_audio_ref: list[float] = [0.0]
    total_proc_ref: list[float] = [0.0]
    total_ttft_ref: list[float] = [0.0]

    # Time-sender thread
    def _send_time() -> None:
        while True:
            if duration is not None:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    break
            time.sleep(1)
            wall = time.time() - start_time
            msg: dict[str, Any] = {
                "type": "time",
                "id": worker_id,
                "wall": format_time(wall),
            }
            if total_proc_ref[0] > 0:
                msg["avg_rtf"] = round(total_proc_ref[0] / total_audio_ref[0], 3) if total_audio_ref[0] > 0 else 0
                msg["avg_tok"] = round(total_tokens_ref[0] / wall, 1) if wall > 0 else 0
            q.put(msg)

    time_thread = __import__("threading").Thread(target=_send_time, daemon=True)
    time_thread.start()

    start_event.wait()

    try:
        while duration is None or (time.time() - start_time < duration):
            if not audio_tasks:
                break

            # Выбираем задачу
            if mode == "once":
                if task_index >= len(audio_tasks):
                    break
                task = audio_tasks[task_index]
            else:
                task = audio_tasks[task_index % len(audio_tasks)]

            audio_path = task["path"]
            audio_dur = task["duration"]

            q.put({
                "type": "live",
                "id": worker_id,
                "tail": f"Processing {Path(audio_path).stem} ({audio_dur:.1f}s)...",
                "wall": format_time(time.time() - start_time),
            })

            processing_start = time.time()
            first_token_time = None
            transcription_text = ""
            token_count = 0

            try:
                with open(audio_path, "rb") as f:
                    kwargs: Dict[str, Any] = {
                        "file": (Path(audio_path).name, f),
                        "model": MODEL,
                        "response_format": response_format,
                    }
                    if prompt:
                        kwargs["prompt"] = prompt
                    if language:
                        kwargs["language"] = language

                    response = client.audio.transcriptions.create(**kwargs)

                # Извлекаем текст транскрипции
                if hasattr(response, "text"):
                    transcription_text = response.text
                elif hasattr(response, "get"):
                    transcription_text = response.get("text", "")
                else:
                    # JSON response
                    if isinstance(response, str):
                        try:
                            parsed = json.loads(response)
                            transcription_text = parsed.get("text", str(response))
                        except json.JSONDecodeError:
                            transcription_text = str(response)
                    else:
                        transcription_text = str(response)

                processing_time = time.time() - processing_start
                token_count = len(transcription_text.split())
                first_token_time = processing_time  # non-streaming: all at once

            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                if not skip_errors:
                    q.put({
                        "type": "error_stop",
                        "id": worker_id,
                        "error": error_msg,
                        "calls": call_count,
                        "wall": format_time(time.time() - start_time),
                    })
                    break
                transcription_text = f"[error] {error_msg[:60]}"
                processing_time = time.time() - processing_start
                token_count = 0
                first_token_time = processing_time

            call_count += 1
            wall_total = time.time() - start_time
            file_stem = Path(audio_path).stem

            # RTF для этого вызова
            rtf = processing_time / audio_dur if audio_dur > 0 else 0

            total_tokens += token_count
            total_audio_seconds += audio_dur
            total_processing_time += processing_time
            if first_token_time is not None:
                total_ttft += first_token_time
                ttft_count += 1

            total_tokens_ref[0] = total_tokens
            total_audio_ref[0] = total_audio_seconds
            total_proc_ref[0] = total_processing_time
            total_ttft_ref[0] = total_ttft

            # Средний RTF
            avg_rtf = total_processing_time / total_audio_seconds if total_audio_seconds > 0 else 0
            # Скорость токенов
            tok_speed = total_tokens / wall_total if wall_total > 0 else 0
            # Аудио скорость (сек аудио / сек wall)
            audio_speed = total_audio_seconds / wall_total if wall_total > 0 else 0

            tail_text = transcription_text[:response_width] if transcription_text else ""

            q.put({
                "type": "stats",
                "id": worker_id,
                "calls": call_count,
                "tokens": total_tokens,
                "call_tokens": token_count,
                "audio_dur": round(audio_dur, 2),
                "proc_time": round(processing_time, 3),
                "rtf": round(rtf, 3),
                "avg_rtf": round(avg_rtf, 3),
                "tok_speed": round(tok_speed, 1),
                "audio_speed": round(audio_speed, 2),
                "ttft": round(first_token_time, 3) if first_token_time else 0,
                "ttft_sum": round(total_ttft, 2),
                "wall": format_time(wall_total),
                "tail": tail_text,
                "file": file_stem,
            })

            task_index += 1

    except Exception:
        q.put({
            "type": "error",
            "id": worker_id,
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Live Table
# ---------------------------------------------------------------------------

class TranscriptionLiveTable:
    """Rich Live таблица для бенчмарка транскрипции ASR."""

    def __init__(
        self,
        duration: Optional[int],
        total_workers: int,
        response_width: int = 60,
    ) -> None:
        self.duration = duration
        self.total_workers = total_workers
        self.response_width = response_width
        self.workers: dict[int, dict[str, Any]] = {}
        self._errors: dict[int, str] = {}
        self.console = Console()

    def mark_started(self, worker_id: int, task_count: int = 0) -> None:
        self.workers[worker_id] = {
            "calls": 0,
            "tokens": 0,
            "call_tokens": 0,
            "audio_dur": 0,
            "proc_time": 0,
            "rtf": 0,
            "avg_rtf": 0,
            "tok_speed": 0,
            "audio_speed": 0,
            "ttft": 0,
            "ttft_sum": 0,
            "wall": "",
            "tail": "[dim]waiting...[/]",
            "file": "",
            "total_tasks": task_count,
        }

    def update_stats(self, msg: Dict[str, Any]) -> None:
        w = self.workers.setdefault(msg["id"], {})
        for k, v in msg.items():
            if k not in ("type", "id") and v is not None:
                w[k] = v

    def update_time(self, msg: Dict[str, Any]) -> None:
        w = self.workers.setdefault(msg["id"], {})
        w["wall"] = msg.get("wall", w.get("wall", ""))
        if "avg_rtf" in msg:
            w["avg_rtf"] = msg["avg_rtf"]
        if "avg_tok" in msg:
            w["tok_speed"] = msg["avg_tok"]

    def mark_error(self, worker_id: int, traceback_str: str) -> None:
        self._errors[worker_id] = traceback_str

    def mark_stopped(self, worker_id: int, error_msg: str) -> None:
        w = self.workers.setdefault(worker_id, {})
        w["stopped"] = True
        w["error"] = error_msg

    def render(self) -> Table:
        table = Table(
            box=_rich_box.DOUBLE,
            show_header=True,
        )

        table.add_column("W#", style="cyan", width=4, justify="right")
        table.add_column("Calls", style="magenta", width=6, justify="right")
        table.add_column("File", style="green", width=14)
        table.add_column("Audio", style="yellow", width=8, justify="right")
        table.add_column("Proc", style="white", width=8, justify="right")
        table.add_column("RTF", style="bold red", width=8, justify="right")
        table.add_column("Avg RTF", style="bold red", width=9, justify="right")
        table.add_column("Tok/s", style="bold green", width=9, justify="right")
        table.add_column("Audio/s", style="bold blue", width=9, justify="right")
        table.add_column("TTFT", style="dim", width=8, justify="right")
        table.add_column("Wall", style="dim", width=7, justify="right")
        table.add_column("Tokens", style="yellow", width=8, justify="right")
        table.add_column("Transcript", style="dim white", width=self.response_width)

        title_parts = ["ASR Transcription Benchmark"]
        if self.duration:
            title_parts.append(f"(duration: {self.duration}s)")
        table.title = " ".join(title_parts)

        for wid in sorted(self.workers.keys()):
            w = self.workers[wid]
            err = self._errors.get(wid)

            if err:
                table.add_row(
                    str(wid), str(w.get("calls", 0)), "", "",
                    "ERROR", "", "", "", "", "", "", "",
                    "[red]SEE LOG[/]",
                )
                continue

            if w.get("stopped"):
                error_text = w.get("error", "unknown")[:self.response_width]
                table.add_row(
                    str(wid), str(w.get("calls", 0)), "", "",
                    "STOPPED", "", "", "", "", "", "", "",
                    f"[red]{error_text}[/]",
                )
                continue

            # Clean tail
            tail = w.get("tail", "")
            if tail and len(tail) > self.response_width:
                tail = tail[:self.response_width - 3] + "..."

            table.add_row(
                str(wid),
                str(w.get("calls", 0)),
                w.get("file", ""),
                f"{w.get('audio_dur', 0):.1f}s",
                f"{w.get('proc_time', 0):.3f}s",
                f"{w.get('rtf', 0):.3f}",
                f"{w.get('avg_rtf', 0):.3f}",
                f"{w.get('tok_speed', 0):.1f}",
                f"{w.get('audio_speed', 0):.2f}x",
                f"{w.get('ttft', 0):.3f}s",
                w.get("wall", ""),
                f"{w.get('tokens', 0):,}",
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
    audio_dir: Optional[str] = None,
    durations: Optional[List[float]] = None,
    max_duration: Optional[float] = None,
    model_timeout: float = 600.0,
    prompt: Optional[str] = None,
    language: Optional[str] = None,
    response_format: str = "json",
    skip_errors: bool = False,
    response_width: int = 60,
    mode: str = "cycling",
) -> None:
    """Запускает бенчмарк транскрипции ASR.

    Args:
        workers: Количество параллельных воркеров.
        duration: Общее время бенчмарка в секундах.
        base_url: URL API (OpenAI-compatible).
        api_key: API ключ.
        model: Название модели.
        audio_dir: Директория с аудио файлами.
        durations: Список длительностей для генерации аудио (сек).
        max_duration: Максимальная длительность аудио (фильтр).
        model_timeout: Таймаут запроса к модели.
        prompt: Промпт для контекста транскрипции.
        language: Язык транскрипции.
        response_format: Формат ответа API.
        skip_errors: Продолжать после ошибки.
        response_width: Ширина колонки Transcript.
        mode: "cycling" (зацикливание) или "once" (один проход).
    """
    # Читаем TRANSCRIPTION_* из os.environ (загружено из .env через utils.py)
    # Приоритет: CLI > TRANSCRIPTION_* env vars > BASE_URL/API_KEY/MODEL из .env
    if base_url is None:
        base_url = os.environ.get("TRANSCRIPTION_BASE_URL")
    if api_key is None:
        api_key = os.environ.get("TRANSCRIPTION_API_KEY")
    if model is None:
        model = os.environ.get("TRANSCRIPTION_MODEL")

    apply_config(base_url=base_url, api_key=api_key, model=model)

    # Экспортируем финальные значения в os.environ, чтобы воркеры (multiprocessing)
    # прочитали их при своём load_dotenv() из utils.py (без override=True
    # существующие os.environ не перезаписываются)
    if base_url:
        os.environ["BASE_URL"] = base_url
    if api_key:
        os.environ["API_KEY"] = api_key
    if model:
        os.environ["MODEL"] = model

    # --- Подготовка аудио ---
    audio_tasks: List[Dict[str, Any]] = []

    if durations:
        # Генерируем аудио указанных длительностей
        print(f"Генерация аудио: {len(durations)} длительностей...")
        series = generate_duration_series(durations)
        audio_tasks = [{"path": str(p), "duration": d} for p, d in series]
        print(f"Сгенерировано {len(audio_tasks)} файлов")
    elif audio_dir:
        # Используем пользовательские файлы
        found = discover_audio_files(audio_dir)
        found = filter_by_max_duration(found, max_duration)
        if not found:
            print(f"Ошибка: аудио не найдено в {audio_dir}")
            sys.exit(1)
        for p in found:
            dur = get_audio_duration(p)
            audio_tasks.append({"path": str(p), "duration": dur})
        print(f"Найдено {len(audio_tasks)} аудио в {audio_dir}")
    else:
        # Используем бандл
        bundled = load_bundled_audio()
        bundled = filter_by_max_duration(bundled, max_duration)
        if not bundled:
            print("Ошибка: аудиофайлы бандла не найдены")
            sys.exit(1)
        for p in bundled:
            dur = get_audio_duration(p)
            audio_tasks.append({"path": str(p), "duration": dur})
        print(f"Бандл: {len(audio_tasks)} аудио")

    if not audio_tasks:
        print("Ошибка: нет аудио файлов для бенчмарка")
        sys.exit(1)

    # --- Заголовок ---
    print("=" * 70)
    print("  ASR Transcription Benchmark")
    print("=" * 70)
    print(f"  Модель:         {model or os.environ.get('MODEL', 'N/A')}")
    print(f"  Воркеров:       {workers}")
    print(f"  Аудио задач:    {len(audio_tasks)}")
    print(f"  Режим:          {mode}")
    print(f"  Timeout:        {model_timeout}s")
    if language:
        print(f"  Язык:           {language}")
    if max_duration:
        print(f"  Max duration: {max_duration}s")
    if durations:
        print(f"  Длительности:   {durations}")
    if duration:
        print(f"  Бенчмарк:       {duration}с")
    print("=" * 70)
    print()

    # --- Запуск воркеров ---
    q: Queue = Queue()  # type: ignore[valid-type]
    start_event: Event = Event()  # type: ignore[valid-type]
    processes: List[Process] = []

    for i in range(workers):
        p = Process(
            target=_worker,
            args=(
                i, q, start_event, audio_tasks, duration, model_timeout,
                prompt, language, response_format, skip_errors, response_width, mode,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Live table ---
    table = TranscriptionLiveTable(duration, workers, response_width)
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

            if duration is not None and mode == "cycling":
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
                table.mark_started(msg["id"], msg.get("tasks", 0))
            elif msg_type == "stats":
                table.update_stats(msg)
            elif msg_type == "time":
                table.update_time(msg)
            elif msg_type == "live":
                table.update_stats(msg)
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
    print("  ИТОГИ (ASR Transcription Benchmark)")
    print("=" * 70)

    for wid in sorted(table.workers.keys()):
        w = table.workers[wid]
        err = table._errors.get(wid)
        if err:
            print(f"  Воркер {wid}: ОШИБКА")
            print(f"    {err[:200]}")
            continue
        print(f"  Воркер {wid}:")
        print(f"    Вызовов:      {w.get('calls', 0)}")
        print(f"    Токенов:      {w.get('tokens', 0):,}")
        print(f"    Avg RTF:      {w.get('avg_rtf', 0):.3f}")
        print(f"    Tok/s:        {w.get('tok_speed', 0):.1f}")
        print(f"    Audio/s:      {w.get('audio_speed', 0):.2f}x")
        print(f"    TTFT sum:     {w.get('ttft_sum', 0):.1f}s")
        print(f"    Wall time:    {w.get('wall', 'N/A')}")

    total_calls = sum(w.get("calls", 0) for w in table.workers.values())
    total_tokens = sum(w.get("tokens", 0) for w in table.workers.values())
    print(f"  Всего вызовов: {total_calls}")
    print(f"  Всего токенов: {total_tokens:,}")
    print("=" * 70)


def cli() -> None:
    """CLI entry point для bench_transcription."""
    parser = argparse.ArgumentParser(
        prog="bench_transcription",
        description="Бенчмарк скорости ASR-моделей (транскрипция речи)",
    )
    add_common_args(parser)
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Количество параллельных воркеров (по умолчанию: 4)",
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=None,
        help="Длительность бенчмарка в секундах (без параметра -- без ограничения)",
    )
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Директория с аудио файлами для теста",
    )
    parser.add_argument(
        "--durations", type=float, nargs="+", default=None,
        help="Список длительностей аудио для генерации (сек). Например: 5 10 30 60",
    )
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Максимальная длительность аудио в секундах (фильтр)",
    )
    parser.add_argument(
        "--model-timeout", type=float, default=None,
        help="Таймаут запроса к модели в секундах (по умолчанию: из TRANSCRIPTION_MODEL_TIMEOUT или 600)",
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default=None,
        help="Промпт для контекста транскрипции",
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None,
        help="Язык транскрипции (например: ru, en)",
    )
    parser.add_argument(
        "--response-format", type=str, default="json",
        choices=["json", "text", "srt", "verbose_json", "vtt"],
        help="Формат ответа API (по умолчанию: json)",
    )
    parser.add_argument(
        "--response-width", type=int, default=60,
        help="Ширина колонки Transcript в таблице (по умолчанию: 60)",
    )
    parser.add_argument(
        "--skip-errors", action="store_true", default=False,
        help="Продолжать после ошибки (по умолчанию воркер останавливается)",
    )
    parser.add_argument(
        "--mode", type=str, default="cycling", choices=["cycling", "once"],
        help="Режим: cycling (зацикливание аудио) или once (один проход)",
    )

    args = parser.parse_args()

    # Загрузка TRANSCRIPTION_MODEL_TIMEOUT если не указан через CLI
    model_timeout = args.model_timeout
    if model_timeout is None:
        model_timeout = float(os.environ.get("TRANSCRIPTION_MODEL_TIMEOUT", "600"))

    run_benchmark(
        workers=args.workers,
        duration=args.duration,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        audio_dir=args.audio,
        durations=args.durations,
        max_duration=args.max_duration,
        model_timeout=model_timeout,
        prompt=args.prompt,
        language=args.language,
        response_format=args.response_format,
        skip_errors=args.skip_errors,
        response_width=args.response_width,
        mode=args.mode,
    )


if __name__ == "__main__":
    cli()