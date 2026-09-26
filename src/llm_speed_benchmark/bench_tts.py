"""
llm_speed_benchmark/bench_tts.py

Бенчмарк скорости TTS-моделей (Text-to-Speech).
Тестирует модели синтеза речи: Qwen3-TTS и другие через OpenAI-compatible API.

Endpoint: audio.speech.create (POST /v1/audio/speech).
Input: текст (str). Output: аудио (бинарный WAV).

Метрики:
  - TTFT (Time To First Token/byte): время до первого байта аудио
  - Speed (char/s): скорость генерации (символы текста / wall time)
  - Speed (audio/s): секунд аудио / секунд wall time (если доступна длительность)
  - Size (bytes): размер сгенерированного аудио
"""

from __future__ import annotations

import argparse
import io
import os
import signal
import sys
import time
import traceback
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.table import Table, box as _rich_box

from .cli_common import add_common_args, apply_config
from .utils import format_time


# ---------------------------------------------------------------------------
# Текстовые промпты разной длины для TTS
# ---------------------------------------------------------------------------

# Короткие фразы (1-3 слова, ~10-50 chars)
SHORT_TEXTS: List[str] = [
    "Привет.",
    "Тест.",
    "Один.",
    "Да.",
    "Нет.",
    "Мир.",
    "Код.",
    "Бенчмарк.",
    "Скорость.",
    "Тест TTS.",
]

# Средние предложения (5-15 слов, ~50-200 chars)
MEDIUM_TEXTS: List[str] = [
    "Это тестовое сообщение для проверки синтеза речи.",
    "Сегодня хорошая погода для прогулки в парке.",
    "Квантовые вычисления открывают новые возможности.",
    "Искусственный интеллект меняет мир вокруг нас.",
    "Разработка программного обеспечения требует внимания.",
    "Машинное обучение позволяет решать сложные задачи.",
    "Нейронные сети обучаются на больших данных.",
    "Автоматизация процессов ускоряет разработку.",
    "Тестирование производительности моделей важно.",
    "Синтез речи из текста это сложная задача.",
]

# Длинные тексты (20-50 слов, ~200-500 chars)
LONG_TEXTS: List[str] = [
    (
        "Привет! Это тестовый синтез речи для проверки работы TTS модели. "
        "Мы тестируем скорость генерации аудио из текста. "
        "Этот текст содержит несколько предложений для оценки производительности."
    ),
    (
        "Бенчмарк скорости TTS моделей включает измерение времени до первого байта, "
        "скорости генерации символов в секунду, и размера полученного аудио файла. "
        "Результаты собираются в многопоточном режиме для статистической достоверности."
    ),
    (
        "Тестирование систем синтеза речи важно для выбора оптимальной модели. "
        "Разные модели показывают разную скорость и качество генерации. "
        "Для приложений реального времени важна низкая задержка первого ответа."
    ),
    (
        "Данный бенчмарк отправляет тексты разной длины на TTS модель через OpenAI "
        "совместимый API. Измеряется время генерации, размер выходного аудио, "
        "и скорость обработки текста в символах в секунду."
    ),
    (
        "В современном мире синтез речи используется в голосовых помощниках, "
        "системах озвучивания текста, навигационных приложениях и доступных "
        "интерфейсах для людей с ограничениями зрения. Качество синтеза постоянно "
        "улучшается благодаря развитию нейросетевых архитектур."
    ),
]

# Очень длинные тексты (50+ слов, ~500+ chars)
VERY_LONG_TEXTS: List[str] = [
    (
        "Это очень длинный текст для проверки производительности системы синтеза речи "
        "на больших входных данных. Мы хотим убедиться, что модель стабильно работает "
        "с текстами разной длины и показывает предсказуемое время генерации. "
        "Тестирование включает несколько повторений для каждого размера текста. "
        "Результаты усредняются для получения достоверных метрик производительности. "
        "Данные собираются в многопроцессном режиме для имитации реальной нагрузки."
    ),
    (
        "Для тестирования TTS моделей мы используем несколько параллельных воркеров, "
        "каждый из которых отправляет текстовые запросы на сервер. "
        "Измеряются следующие метрики: время до первого байта аудио, общее время "
        "генерации, размер полученного аудио файла в байтах, и скорость генерации "
        "в символах текста в секунду. Эти метрики позволяют оценить производительность "
        "модели для различных сценариев использования."
    ),
]

# TODO: можно добавить случайную генерацию промптов для избегания кэширования
TTS_PROMPTS: Dict[str, List[str]] = {
    "short": SHORT_TEXTS,
    "medium": MEDIUM_TEXTS,
    "long": LONG_TEXTS,
    "very_long": VERY_LONG_TEXTS,
}

# Длины текстов в символах для категорий
TEXT_LENGTH_BINS: List[Tuple[str, int, int]] = [
    ("short", 10, 50),
    ("medium", 50, 200),
    ("long", 200, 500),
    ("very_long", 500, 2000),
]


def get_texts_by_char_count(texts: List[str], min_chars: int = 10, max_chars: int = 500) -> List[str]:
    """Фильтрует тексты по диапазону длины в символах.

    Args:
        texts: Список текстов.
        min_chars: Минимальное количество символов.
        max_chars: Максимальное количество символов.

    Returns:
        Отфильтрованный список текстов.
    """
    return [t for t in texts if min_chars <= len(t) <= max_chars]


def build_text_tasks(
    char_counts: Optional[List[int]] = None,
    texts_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Строит список задач для TTS бенчмарка.

    Args:
        char_counts: Список длин текстов в символах для генерации.
            Если None, использует тексты из TTS_PROMPTS.
        texts_dir: Директория с текстовыми файлами (не реализовано).

    Returns:
        Список задач: [{"text": str, "char_count": int}, ...].
    """
    tasks: List[Dict[str, Any]] = []

    if texts_dir:
        # TODO: загружать тексты из директории
        print(f"Предупреждение: загрузка текстов из директории не реализована")
        pass

    if char_counts:
        # Генерируем тексты указанной длины
        for count in char_counts:
            # Находим подходящую категорию
            matched = False
            for cat, min_c, max_c in TEXT_LENGTH_BINS:
                if min_c <= count <= max_c:
                    pool = TTS_PROMPTS.get(cat, LONG_TEXTS)
                    # Берём первый подходящий по длине или обрезаем/дополняем
                    text = pool[0] if pool else "Тестовый текст."
                    # Обрезаем или повторяем до нужной длины
                    if len(text) < count:
                        repeats = count // len(text) + 1
                        text = (text * repeats)[:count]
                    else:
                        # Ищем текст максимально близкий к нужной длине
                        best = min(pool, key=lambda t: abs(len(t) - count))
                        text = best[:count]
                    tasks.append({"text": text, "char_count": count})
                    matched = True
                    break
            if not matched:
                # Сверхдлинный текст
                base = VERY_LONG_TEXTS[0] if VERY_LONG_TEXTS else "Тест."
                repeats = count // len(base) + 1
                text = (base * repeats)[:count]
                tasks.append({"text": text, "char_count": count})
    else:
        # Используем все доступные тексты
        for cat, texts in TTS_PROMPTS.items():
            for t in texts:
                tasks.append({"text": t, "char_count": len(t)})

    return tasks


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(  # type: ignore[valid-type]
    worker_id: int,
    q: Queue,  # type: ignore[valid-type]
    start_event: Event,  # type: ignore[valid-type]
    text_tasks: List[Dict[str, Any]],
    duration: Optional[int],
    model_timeout: float,
    voice: str,
    response_format: str,
    skip_errors: bool,
    response_width: int,
    mode: str,
) -> None:
    """Worker для бенчмарка TTS.

    Args:
        worker_id: ID воркера.
        q: Очередь сообщений.
        start_event: Событие синхронизации старта.
        text_tasks: Список задач с текстом.
        duration: Общее время бенчмарка (None = без лимита).
        model_timeout: Таймаут запроса к модели.
        voice: Голос TTS.
        response_format: Формат аудио (wav, mp3 и т.д.).
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
        "tasks": len(text_tasks),
    })

    start_time = time.time()
    call_count = 0
    total_chars = 0
    total_size = 0
    total_processing_time = 0.0
    total_ttft = 0.0
    ttft_count = 0
    task_index = 0

    # Mutable refs для live-обновлений
    total_chars_ref: list[int] = [0]
    total_size_ref: list[int] = [0]
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
                msg["avg_proc"] = round(total_proc_ref[0] / call_count, 3) if call_count > 0 else 0
                msg["avg_chars"] = round(total_chars_ref[0] / wall, 1) if wall > 0 else 0
            q.put(msg)

    time_thread = __import__("threading").Thread(target=_send_time, daemon=True)
    time_thread.start()

    start_event.wait()

    try:
        while duration is None or (time.time() - start_time < duration):
            if not text_tasks:
                break

            # Выбираем задачу
            if mode == "once":
                if task_index >= len(text_tasks):
                    break
                task = text_tasks[task_index]
            else:
                task = text_tasks[task_index % len(text_tasks)]

            text = task["text"]
            char_count = task["char_count"]

            q.put({
                "type": "live",
                "id": worker_id,
                "tail": f"Processing {char_count} chars...",
                "wall": format_time(time.time() - start_time),
            })

            processing_start = time.time()
            first_byte_time = None
            audio_data = b""

            try:
                # OpenAI TTS API: client.audio.speech.create()
                # Возвращает streaming response, читаем всё
                response = client.audio.speech.create(
                    model=MODEL,
                    voice=voice,
                    input=text,
                    response_format=response_format,
                )

                # Читаем все байты
                if hasattr(response, "read"):
                    audio_data = response.read()
                elif hasattr(response, "content"):
                    audio_data = response.content
                elif hasattr(response, "iter_bytes"):
                    audio_data = b"".join(response.iter_bytes())
                else:
                    # stream_to_file или другой метод
                    buf = io.BytesIO()
                    for chunk in response:  # type: ignore[attr-defined]
                        buf.write(chunk)
                    audio_data = buf.getvalue()

                processing_time = time.time() - processing_start
                first_byte_time = processing_time  # non-streaming: all at once

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
                audio_data = b"[error]"
                processing_time = time.time() - processing_start

            call_count += 1
            wall_total = time.time() - start_time

            total_chars += char_count
            total_size += len(audio_data)
            total_processing_time += processing_time
            if first_byte_time is not None:
                total_ttft += first_byte_time
                ttft_count += 1

            total_chars_ref[0] = total_chars
            total_size_ref[0] = total_size
            total_proc_ref[0] = total_processing_time
            total_ttft_ref[0] = total_ttft

            # Метрики
            avg_proc = total_processing_time / call_count if call_count > 0 else 0
            char_speed = total_chars / wall_total if wall_total > 0 else 0
            size_speed = total_size / wall_total if wall_total > 0 else 0

            tail_text = text[:response_width] if text else ""

            q.put({
                "type": "stats",
                "id": worker_id,
                "calls": call_count,
                "chars": total_chars,
                "call_chars": char_count,
                "size": len(audio_data),
                "total_size": total_size,
                "proc_time": round(processing_time, 3),
                "avg_proc": round(avg_proc, 3),
                "char_speed": round(char_speed, 1),
                "size_speed": round(size_speed / 1024, 1),  # KB/s
                "ttft": round(first_byte_time, 3) if first_byte_time else 0,
                "ttft_sum": round(total_ttft, 2),
                "wall": format_time(wall_total),
                "tail": tail_text,
                "char_count": char_count,
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

class TTSLiveTable:
    """Rich Live таблица для бенчмарка TTS."""

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
            "chars": 0,
            "call_chars": 0,
            "size": 0,
            "total_size": 0,
            "proc_time": 0,
            "avg_proc": 0,
            "char_speed": 0,
            "size_speed": 0,
            "ttft": 0,
            "ttft_sum": 0,
            "wall": "",
            "tail": "[dim]waiting...[/]",
            "char_count": 0,
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
        if "avg_proc" in msg:
            w["avg_proc"] = msg["avg_proc"]
        if "avg_chars" in msg:
            w["char_speed"] = msg["avg_chars"]

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
        table.add_column("Chars", style="green", width=8, justify="right")
        table.add_column("Size", style="yellow", width=10, justify="right")
        table.add_column("Proc", style="white", width=8, justify="right")
        table.add_column("Avg P.", style="bold red", width=8, justify="right")
        table.add_column("Ch/s", style="bold green", width=9, justify="right")
        table.add_column("KB/s", style="bold blue", width=9, justify="right")
        table.add_column("TTFT", style="dim", width=8, justify="right")
        table.add_column("Wall", style="dim", width=7, justify="right")
        table.add_column("Text", style="dim white", width=self.response_width)

        title_parts = ["TTS Benchmark"]
        if self.duration:
            title_parts.append(f"(duration: {self.duration}s)")
        table.title = " ".join(title_parts)

        for wid in sorted(self.workers.keys()):
            w = self.workers[wid]
            err = self._errors.get(wid)

            if err:
                table.add_row(
                    str(wid), str(w.get("calls", 0)), "", "",
                    "ERROR", "", "", "", "", "",
                    "[red]SEE LOG[/]",
                )
                continue

            if w.get("stopped"):
                error_text = w.get("error", "unknown")[:self.response_width]
                table.add_row(
                    str(wid), str(w.get("calls", 0)), "", "",
                    "STOPPED", "", "", "", "", "",
                    f"[red]{error_text}[/]",
                )
                continue

            size_kb = w.get("size", 0) / 1024
            total_size_kb = w.get("total_size", 0) / 1024

            tail = w.get("tail", "")
            if tail and len(tail) > self.response_width:
                tail = tail[:self.response_width - 3] + "..."

            table.add_row(
                str(wid),
                str(w.get("calls", 0)),
                f"{w.get('chars', 0):,}",
                f"{total_size_kb:.1f}KB",
                f"{w.get('proc_time', 0):.3f}s",
                f"{w.get('avg_proc', 0):.3f}s",
                f"{w.get('char_speed', 0):.1f}",
                f"{w.get('size_speed', 0):.1f}",
                f"{w.get('ttft', 0):.3f}s",
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
    char_counts: Optional[List[int]] = None,
    voice: str = "agata",
    response_format: str = "wav",
    model_timeout: float = 600.0,
    skip_errors: bool = False,
    response_width: int = 60,
    mode: str = "cycling",
) -> None:
    """Запускает бенчмарк TTS.

    Args:
        workers: Количество параллельных воркеров.
        duration: Общее время бенчмарка в секундах.
        base_url: URL API (OpenAI-compatible).
        api_key: API ключ.
        model: Название модели.
        char_counts: Список длин текстов в символах.
        voice: Голос TTS.
        response_format: Формат аудио (wav, mp3...).
        model_timeout: Таймаут запроса к модели.
        skip_errors: Продолжать после ошибки.
        response_width: Ширина колонки Text.
        mode: "cycling" или "once".
    """
    # Читаем TTS_* из os.environ
    if base_url is None:
        base_url = os.environ.get("TTS_BASE_URL")
    if api_key is None:
        api_key = os.environ.get("TTS_API_KEY")
    if model is None:
        model = os.environ.get("TTS_MODEL")

    apply_config(base_url=base_url, api_key=api_key, model=model)

    # Экспортируем для воркеров
    if base_url:
        os.environ["BASE_URL"] = base_url
    if api_key:
        os.environ["API_KEY"] = api_key
    if model:
        os.environ["MODEL"] = model

    # --- Подготовка текстовых задач ---
    text_tasks = build_text_tasks(char_counts=char_counts)

    if not text_tasks:
        print("Ошибка: нет текстов для бенчмарка")
        sys.exit(1)

    print(f"Подготовлено {len(text_tasks)} текстовых задач")
    if char_counts:
        print(f"Длины текстов (chars): {char_counts}")

    # --- Заголовок ---
    print("=" * 70)
    print("  TTS Benchmark")
    print("=" * 70)
    print(f"  Модель:         {model or os.environ.get('MODEL', 'N/A')}")
    print(f"  Воркеров:       {workers}")
    print(f"  Текстов:        {len(text_tasks)}")
    print(f"  Режим:          {mode}")
    print(f"  Timeout:        {model_timeout}s")
    print(f"  Голос:          {voice}")
    print(f"  Формат:         {response_format}")
    if char_counts:
        print(f"  Chars:          {char_counts}")
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
                i, q, start_event, text_tasks, duration, model_timeout,
                voice, response_format, skip_errors, response_width, mode,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Live table ---
    table = TTSLiveTable(duration, workers, response_width)
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
    print("  ИТОГИ (TTS Benchmark)")
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
        print(f"    Символов:     {w.get('chars', 0):,}")
        print(f"    Аудио:        {w.get('total_size', 0) / 1024:.1f}KB")
        print(f"    Avg time:     {w.get('avg_proc', 0):.3f}s")
        print(f"    Char/s:       {w.get('char_speed', 0):.1f}")
        print(f"    KB/s:         {w.get('size_speed', 0):.1f}")
        print(f"    TTFT sum:     {w.get('ttft_sum', 0):.1f}s")
        print(f"    Wall time:    {w.get('wall', 'N/A')}")

    total_calls = sum(w.get("calls", 0) for w in table.workers.values())
    total_chars = sum(w.get("chars", 0) for w in table.workers.values())
    print(f"  Всего вызовов: {total_calls}")
    print(f"  Всего символов: {total_chars:,}")
    print("=" * 70)


def cli() -> None:
    """CLI entry point для bench_tts."""
    parser = argparse.ArgumentParser(
        prog="bench_tts",
        description="Бенчмарк скорости TTS-моделей (синтез речи из текста)",
    )
    add_common_args(parser)
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Количество параллельных воркеров (по умолчанию: 4)",
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=None,
        help="Длительность бенчмарка в секундах",
    )
    parser.add_argument(
        "--char-counts", type=int, nargs="+", default=None,
        help="Список длин текстов в символах. Например: 10 50 200 500",
    )
    parser.add_argument(
        "--voice", type=str, default="agata",
        help="Голос TTS (по умолчанию: agata)",
    )
    parser.add_argument(
        "--response-format", type=str, default="wav",
        choices=["wav", "mp3", "opus", "pcm"],
        help="Формат аудио (по умолчанию: wav)",
    )
    parser.add_argument(
        "--model-timeout", type=float, default=None,
        help="Таймаут запроса к модели в секундах (по умолчанию: из TTS_MODEL_TIMEOUT или 600)",
    )
    parser.add_argument(
        "--response-width", type=int, default=60,
        help="Ширина колонки Text в таблице (по умолчанию: 60)",
    )
    parser.add_argument(
        "--skip-errors", action="store_true", default=False,
        help="Продолжать после ошибки",
    )
    parser.add_argument(
        "--mode", type=str, default="cycling", choices=["cycling", "once"],
        help="Режим: cycling (зацикливание) или once (один проход)",
    )

    args = parser.parse_args()

    model_timeout = args.model_timeout
    if model_timeout is None:
        model_timeout = float(os.environ.get("TTS_MODEL_TIMEOUT", "600"))

    run_benchmark(
        workers=args.workers,
        duration=args.duration,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        char_counts=args.char_counts,
        voice=args.voice,
        response_format=args.response_format,
        model_timeout=model_timeout,
        skip_errors=args.skip_errors,
        response_width=args.response_width,
        mode=args.mode,
    )


if __name__ == "__main__":
    cli()