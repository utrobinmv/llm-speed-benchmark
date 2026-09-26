"""
llm_speed_benchmark/bench_emb.py

Бенчмарк скорости эмбединг-моделей (Embedding).
Тестирует модели: embeddinggemma-300m и другие через OpenAI-compatible API.

Endpoint: embeddings.create (POST /v1/embeddings).
Input: текст (строка или массив строк). Output: вектор эмбединга.

Метрики:
  - TTFT: время до получения эмбединга
  - Speed (docs/s): количество документов в секунду
  - Speed (chars/s): символов текста в секунду
  - Dimension: размерность эмбединга
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
from .utils import format_time


# ---------------------------------------------------------------------------
# Текстовые запросы разной длины для эмбединга
# ---------------------------------------------------------------------------

SHORT_TEXTS: List[str] = [
    "Привет.",
    "Тест эмбединга.",
    "Короткий текст.",
    "Одно слово.",
    "Бенчмарк.",
    "Скорость.",
    "Вектор.",
    "Модель.",
    "Данные.",
    "Запрос.",
]

MEDIUM_TEXTS: List[str] = [
    "Это средний текст для проверки эмбединг модели.",
    "Эмбединги представляют текст в виде вектора чисел.",
    "Машинное обучение использует эмбединги для анализа.",
    "Поиск по сходству основан на косинусной близости.",
    "Кластеризация документов использует эмбединги.",
    "Рекомендательные системы основаны на эмбедингах.",
    "Трансформеры генерируют контекстные эмбединги.",
    "Классификация текстов использует эмбединги.",
    "Семантический поиск использует эмбединги.",
    "Эмбединги фиксированной размерности удобны.",
]

LONG_TEXTS: List[str] = [
    (
        "Эмбединг модель преобразует текст в плотный вектор фиксированной размерности. "
        "Этот вектор представляет семантическое значение текста. "
        "Близкие по смыслу тексты имеют близкие векторы."
    ),
    (
        "Для тестирования производительности эмбединг модели отправляются тексты "
        "разной длины. Измеряется время генерации эмбединга и скорость обработки "
        "в символах в секунду."
    ),
    (
        "Бенчмарк эмбединг моделей включает несколько параллельных воркеров. "
        "Каждый воркер отправляет текстовый запрос через OpenAI совместимый API. "
        "Результаты собираются и усредняются."
    ),
    (
        "Эмбединги широко используются в системах поиска, классификации, "
        "кластеризации и рекомендаций. Размерность эмбединга влияет на точность "
        "и производительность последующих операций."
    ),
    (
        "Качество эмбединга зависит от модели и обучающих данных. "
        "Современные модели генерируют контекстные эмбединги, учитывающие "
        "окружение слов в тексте."
    ),
]

# Тексты с batch-запросами (несколько текстов в одном вызове)
BATCH_TEXTS: List[List[str]] = [
    ["Первый документ.", "Второй документ.", "Третий документ."],
    ["Короткий текст.", "Средний по длине текст для теста.", "Очень длинный текст для проверки производительности эмбединг модели на больших данных."],
    ["Привет.", "Здравствуйте.", "Добрый день.", "Доброе утро.", "Спокойной ночи."],
    [
        "Кошка сидит на окне.",
        "Собака бежит по парку.",
        "Птица летит высоко в небе.",
        "Рыба плавает в пруду.",
        "Лошадь скачет по полю.",
    ],
]


def build_emb_tasks(
    char_counts: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Строит список задач для эмбединг бенчмарка.

    Args:
        char_counts: Список длин текстов в символах для генерации.
            Если None, использует тексты из пулов.
        batch_sizes: Список размеров батча (сколько текстов в одном запросе).
            Если None, каждый запрос содержит один текст.

    Returns:
        Список задач: [{"texts": [str, ...], "char_count": int}, ...].
    """
    tasks: List[Dict[str, Any]] = []
    all_texts = SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS

    if char_counts:
        for count in char_counts:
            if count <= 50:
                pool = SHORT_TEXTS
            elif count <= 200:
                pool = MEDIUM_TEXTS
            else:
                pool = LONG_TEXTS

            base = pool[0] if pool else "Тест."
            if len(base) < count:
                repeats = count // len(base) + 1
                text = (base * repeats)[:count]
            else:
                best = min(pool, key=lambda t: abs(len(t) - count))
                text = best[:count]

            batch = batch_sizes[0] if batch_sizes else 1
            texts = [text]
            # Если batch > 1, добавляем вариации
            for i in range(1, batch):
                src_pool = pool if pool else all_texts
                alt = src_pool[i % len(src_pool)]
                if len(alt) < count:
                    repeats = count // len(alt) + 1
                    alt = (alt * repeats)[:count]
                else:
                    alt = alt[:count]
                texts.append(alt)

            total_chars = sum(len(t) for t in texts)
            tasks.append({
                "texts": texts,
                "char_count": total_chars,
                "batch_size": len(texts),
            })
    else:
        # Одиночные тексты
        for text in all_texts:
            tasks.append({
                "texts": [text],
                "char_count": len(text),
                "batch_size": 1,
            })

        # Batch запросы
        if batch_sizes:
            for batch in BATCH_TEXTS:
                total_chars = sum(len(t) for t in batch)
                tasks.append({
                    "texts": batch,
                    "char_count": total_chars,
                    "batch_size": len(batch),
                })

    return tasks


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(  # type: ignore[valid-type]
    worker_id: int,
    q: Queue,  # type: ignore[valid-type]
    start_event: Event,  # type: ignore[valid-type]
    emb_tasks: List[Dict[str, Any]],
    duration: Optional[int],
    model_timeout: float,
    skip_errors: bool,
    response_width: int,
    mode: str,
) -> None:
    """Worker для бенчмарка эмбедингов.

    Args:
        worker_id: ID воркера.
        q: Очередь сообщений.
        start_event: Событие синхронизации старта.
        emb_tasks: Список задач с текстами.
        duration: Общее время бенчмарка (None = без лимита).
        model_timeout: Таймаут запроса к модели.
        skip_errors: Продолжать после ошибки.
        response_width: Ширина колонки.
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
        "tasks": len(emb_tasks),
    })

    start_time = time.time()
    call_count = 0
    total_chars = 0
    total_docs = 0
    total_processing_time = 0.0
    total_ttft = 0.0
    ttft_count = 0
    task_index = 0

    total_chars_ref: list[int] = [0]
    total_docs_ref: list[int] = [0]
    total_proc_ref: list[float] = [0.0]
    total_ttft_ref: list[float] = [0.0]

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
            q.put(msg)

    time_thread = __import__("threading").Thread(target=_send_time, daemon=True)
    time_thread.start()

    start_event.wait()

    try:
        while duration is None or (time.time() - start_time < duration):
            if not emb_tasks:
                break

            if mode == "once":
                if task_index >= len(emb_tasks):
                    break
                task = emb_tasks[task_index]
            else:
                task = emb_tasks[task_index % len(emb_tasks)]

            texts = task["texts"]
            char_count = task["char_count"]
            batch_size = task.get("batch_size", 1)

            q.put({
                "type": "live",
                "id": worker_id,
                "tail": f"Processing {batch_size} docs ({char_count} chars)...",
                "wall": format_time(time.time() - start_time),
            })

            processing_start = time.time()
            first_chunk_time = None

            try:
                # OpenAI Embedding API: client.embeddings.create()
                response = client.embeddings.create(
                    model=MODEL,
                    input=texts if len(texts) > 1 else texts[0],
                )

                processing_time = time.time() - processing_start
                first_chunk_time = processing_time

                # Размерность эмбединга
                emb_dim = 0
                if response.data and len(response.data) > 0:
                    emb_dim = len(response.data[0].embedding)

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
                processing_time = time.time() - processing_start
                emb_dim = 0

            call_count += 1
            wall_total = time.time() - start_time

            total_chars += char_count
            total_docs += batch_size
            total_processing_time += processing_time
            if first_chunk_time is not None:
                total_ttft += first_chunk_time
                ttft_count += 1

            total_chars_ref[0] = total_chars
            total_docs_ref[0] = total_docs
            total_proc_ref[0] = total_processing_time
            total_ttft_ref[0] = total_ttft

            avg_proc = total_processing_time / call_count if call_count > 0 else 0
            char_speed = total_chars / wall_total if wall_total > 0 else 0
            doc_speed = total_docs / wall_total if wall_total > 0 else 0

            tail_text = f"{texts[0][:response_width]}..." if len(texts[0]) > response_width else texts[0]

            q.put({
                "type": "stats",
                "id": worker_id,
                "calls": call_count,
                "chars": total_chars,
                "call_chars": char_count,
                "docs": total_docs,
                "batch_size": batch_size,
                "proc_time": round(processing_time, 3),
                "avg_proc": round(avg_proc, 3),
                "char_speed": round(char_speed, 1),
                "doc_speed": round(doc_speed, 1),
                "emb_dim": emb_dim,
                "ttft": round(first_chunk_time, 3) if first_chunk_time else 0,
                "ttft_sum": round(total_ttft, 2),
                "wall": format_time(wall_total),
                "tail": tail_text,
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

class EmbeddingLiveTable:
    """Rich Live таблица для бенчмарка эмбедингов."""

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
            "docs": 0,
            "batch_size": 1,
            "proc_time": 0,
            "avg_proc": 0,
            "char_speed": 0,
            "doc_speed": 0,
            "emb_dim": 0,
            "ttft": 0,
            "ttft_sum": 0,
            "wall": "",
            "tail": "[dim]waiting...[/]",
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
        table.add_column("Docs", style="green", width=7, justify="right")
        table.add_column("Chars", style="yellow", width=8, justify="right")
        table.add_column("Dim", style="white", width=6, justify="right")
        table.add_column("Proc", style="bold red", width=8, justify="right")
        table.add_column("Ch/s", style="bold green", width=9, justify="right")
        table.add_column("D/s", style="bold blue", width=9, justify="right")
        table.add_column("TTFT", style="dim", width=8, justify="right")
        table.add_column("Wall", style="dim", width=7, justify="right")
        table.add_column("Input", style="dim white", width=self.response_width)

        title_parts = ["Embedding Benchmark"]
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

            tail = w.get("tail", "")
            if tail and len(tail) > self.response_width:
                tail = tail[:self.response_width - 3] + "..."

            table.add_row(
                str(wid),
                str(w.get("calls", 0)),
                str(w.get("docs", 0)),
                f"{w.get('chars', 0):,}",
                str(w.get("emb_dim", "")),
                f"{w.get('proc_time', 0):.3f}s",
                f"{w.get('char_speed', 0):.1f}",
                f"{w.get('doc_speed', 0):.1f}",
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
    batch_sizes: Optional[List[int]] = None,
    model_timeout: float = 600.0,
    skip_errors: bool = False,
    response_width: int = 60,
    mode: str = "cycling",
) -> None:
    """Запускает бенчмарк эмбедингов.

    Args:
        workers: Количество параллельных воркеров.
        duration: Общее время бенчмарка в секундах.
        base_url: URL API.
        api_key: API ключ.
        model: Название модели.
        char_counts: Список длин текстов в символах.
        batch_sizes: Список размеров батча.
        model_timeout: Таймаут запроса к модели.
        skip_errors: Продолжать после ошибки.
        response_width: Ширина колонки.
        mode: "cycling" или "once".
    """
    if base_url is None:
        base_url = os.environ.get("EMB_BASE_URL")
    if api_key is None:
        api_key = os.environ.get("EMB_API_KEY")
    if model is None:
        model = os.environ.get("EMB_MODEL")

    apply_config(base_url=base_url, api_key=api_key, model=model)

    if base_url:
        os.environ["BASE_URL"] = base_url
    if api_key:
        os.environ["API_KEY"] = api_key
    if model:
        os.environ["MODEL"] = model

    # --- Подготовка задач ---
    emb_tasks = build_emb_tasks(char_counts=char_counts, batch_sizes=batch_sizes)

    if not emb_tasks:
        print("Ошибка: нет текстов для бенчмарка")
        sys.exit(1)

    print(f"Подготовлено {len(emb_tasks)} задач")
    if char_counts:
        print(f"Длины текстов (chars): {char_counts}")
    if batch_sizes:
        print(f"Размеры батча: {batch_sizes}")

    # --- Заголовок ---
    print("=" * 70)
    print("  Embedding Benchmark")
    print("=" * 70)
    print(f"  Модель:         {model or os.environ.get('MODEL', 'N/A')}")
    print(f"  Воркеров:       {workers}")
    print(f"  Задач:          {len(emb_tasks)}")
    print(f"  Режим:          {mode}")
    print(f"  Timeout:        {model_timeout}s")
    if char_counts:
        print(f"  Chars:          {char_counts}")
    if batch_sizes:
        print(f"  Batches:        {batch_sizes}")
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
                i, q, start_event, emb_tasks, duration, model_timeout,
                skip_errors, response_width, mode,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Live table ---
    table = EmbeddingLiveTable(duration, workers, response_width)
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
    print("  ИТОГИ (Embedding Benchmark)")
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
        print(f"    Документов:   {w.get('docs', 0):,}")
        print(f"    Символов:     {w.get('chars', 0):,}")
        print(f"    Размерность:  {w.get('emb_dim', 'N/A')}")
        print(f"    Avg time:     {w.get('avg_proc', 0):.3f}s")
        print(f"    Char/s:       {w.get('char_speed', 0):.1f}")
        print(f"    Doc/s:        {w.get('doc_speed', 0):.1f}")
        print(f"    TTFT sum:     {w.get('ttft_sum', 0):.1f}s")
        print(f"    Wall time:    {w.get('wall', 'N/A')}")

    total_calls = sum(w.get("calls", 0) for w in table.workers.values())
    total_docs = sum(w.get("docs", 0) for w in table.workers.values())
    print(f"  Всего вызовов: {total_calls}")
    print(f"  Всего документов: {total_docs:,}")
    print("=" * 70)


def cli() -> None:
    """CLI entry point для bench_emb."""
    parser = argparse.ArgumentParser(
        prog="bench_emb",
        description="Бенчмарк скорости эмбединг-моделей",
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
        "--batch-sizes", type=int, nargs="+", default=None,
        help="Размеры батча для тестирования. Например: 1 5 10",
    )
    parser.add_argument(
        "--model-timeout", type=float, default=None,
        help="Таймаут запроса к модели (по умолчанию: из EMB_MODEL_TIMEOUT или 600)",
    )
    parser.add_argument(
        "--response-width", type=int, default=60,
        help="Ширина колонки Input в таблице (по умолчанию: 60)",
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
        model_timeout = float(os.environ.get("EMB_MODEL_TIMEOUT", "600"))

    run_benchmark(
        workers=args.workers,
        duration=args.duration,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        char_counts=args.char_counts,
        batch_sizes=args.batch_sizes,
        model_timeout=model_timeout,
        skip_errors=args.skip_errors,
        response_width=args.response_width,
        mode=args.mode,
    )


if __name__ == "__main__":
    cli()