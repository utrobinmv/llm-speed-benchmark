#!/usr/bin/env python3
"""
llm_speed_benchmark/bench_multi.py

Многопроцессный бенчмарк скорости vLLM в режиме стриминга.
Запускает N изолированных процессов, каждый независимо накапливает контекст,
выводит обновляемую таблицу с Rich Live -- без скролла, на одном экране.

Использование:
  bench_multi
  bench_multi --workers 8 --duration 60
"""

import os
import sys
import time
import signal
import argparse
import traceback
from datetime import datetime
from multiprocessing import Process, Queue, Event

from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.box import DOUBLE

from .utils import (
    truncate_history,
    format_time,
    API_KEY,
    BASE_URL,
    MAX_CONTEXT_TOKENS,
    MODEL,
    _token_limit_warn,
)
from .cli_common import add_common_args, apply_config
from .streaming import StreamSession
from .live_table import BaseLiveTable
from .worker_common import make_time_sender, make_on_chunk_callback, send_stats, send_error_stats


def worker(  # type: ignore[reportInvalidTypeForm]
    worker_id: int,
    q: "Queue",
    start_event: "Event",
    duration: int | None,
    initial_messages: list[dict] | None = None,
    skip_errors: bool = False,
) -> None:
    """Запускает цикл вызовов LLM в отдельном процессе.

    Args:
        worker_id: ID воркера.
        q: Queue для сообщений.
        start_event: Event для синхронизации старта.
        duration: Длительность в секундах.
        initial_messages: Начальные сообщения для long context (из датасета).
        skip_errors: Если True — продолжать после ошибки.
    """
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)
        session = StreamSession(client)

        is_long_context = initial_messages is not None and len(initial_messages) > 1

        if is_long_context:
            # Long context: используем messages из датасета
            messages: list[dict] = list(initial_messages)  # type: ignore[arg-type]
            # Помечаем воркер
            q.put({"type": "start", "id": worker_id, "long_context": True})
        else:
            messages = [
                {"role": "system", "content": "Ты полезный помощник. Отвечай подробно и развёрнуто."}
            ]
            q.put({"type": "start", "id": worker_id, "long_context": False})

        start_time = time.time()
        call_count = 0
        total_gen = 0
        total_chunks = 0  # кумулятивный счётчик чанков с момента старта воркера
        prompt_tokens = 0  # берётся из API usage после каждого вызова
        round_num = 1
        total_ttft = 0.0
        ttft_count = 0

        start_event.wait()  # Синхронизация старта

        # --- Mutable references для shared callback'ов ---
        total_gen_ref = [0]
        total_chunks_ref = [0]
        total_ttft_ref = [0.0]

        # --- Поток для обновления wall-time каждую секунду ---
        _time_thread = make_time_sender(q, worker_id, start_time, duration, None, None)

        # --- Callback для live-обновлений во время стриминга ---
        on_chunk = make_on_chunk_callback(q, worker_id, total_gen_ref, total_chunks_ref, total_ttft_ref, None)

        while duration is None or (time.time() - start_time < duration):
            # Уникальные промпты с ID воркера и раундом
            if call_count == 0:
                prompt_text = f"Поток {worker_id:07d}. Что ты умеешь? Расскажи обо всём максимально подробно."
            else:
                prompt_text = f"Поток {worker_id:07d} (раунд {round_num}) продолжай"

            # Проверка лимита контекста -- новый раунд
            if prompt_tokens + total_gen >= MAX_CONTEXT_TOKENS:
                messages = [{"role": "system", "content": "Ты полезный помощник. Отвечай подробно и развёрнуто."}]
                round_num += 1
                call_count = 0
                total_gen = 0
                # total_chunks НЕ сбрасываем -- кумулятивный счётчик с момента старта
                prompt_tokens = 0
                total_ttft = 0.0
                ttft_count = 0

            if prompt_tokens + total_gen >= _token_limit_warn():
                messages = truncate_history(messages)

            messages.append({"role": "user", "content": prompt_text})

            assistant_content = ""
            metrics = None

            try:
                # Настройка callback для live-обновлений
                session.on_chunk = on_chunk
                session.on_chunk_args = {
                    "start_time": start_time,
                }

                # === Streaming через StreamSession ===
                metrics = session.run(
                    messages=messages,
                    model=MODEL,
                )

                assistant_content = metrics.assistant_content

            except Exception as e:  # noqa: BLE001
                q.put({
                    "type": "error_stop",
                    "id": worker_id,
                    "error": str(e),
                    "calls": call_count,
                    "wall": format_time(time.time() - start_time),
                })
                if not skip_errors:
                    break
                assistant_content = f"[red]Error: {e}[/]"

            if assistant_content and not assistant_content.startswith("[red]Error:"):
                messages.append({"role": "assistant", "content": assistant_content})

            call_count += 1
            wall_total = time.time() - start_time

            # Если ошибка в stream -- пропускаем
            if metrics is None:
                send_error_stats(
                    q, worker_id, call_count, total_gen, wall_total,
                    None, 0, assistant_content[:80] if assistant_content else "error",
                )
                continue

            completion_tokens = metrics.completion_tokens
            chunk_count = metrics.chunk_count

            # Защита: пустой ответ
            if completion_tokens == 0:
                send_error_stats(
                    q, worker_id, call_count, total_gen, wall_total,
                    None, 0, "empty",
                )
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

            # Обновляем mutable references для callback'ов
            total_gen_ref[0] = total_gen
            total_chunks_ref[0] = total_chunks
            total_ttft_ref[0] = total_ttft

            # Обновляем prompt_tokens из API (точное значение)
            prompt_tokens = metrics.prompt_tokens

            # Отправляем финальную статистику
            send_stats(
                q, worker_id, call_count, metrics, total_gen, total_chunks, total_ttft,
                wall_total, None, 0, assistant_content[-80:] if assistant_content else "",
                round_num=round_num,
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + total_gen,
            )

    except Exception:  # noqa: BLE001
        # Ловим ошибки на уровне воркера
        q.put({
            "type": "error",
            "id": worker_id,
            "traceback": traceback.format_exc()
        })


# ---------------------------------------------------------------------------
# Rich Live Table
# ---------------------------------------------------------------------------

class LiveTable(BaseLiveTable):
    """Живая таблица для bench_multi с расширенным набором колонок.

    Наследует управление данными из BaseLiveTable (update_stats, update_live,
    update_time, mark_error, mark_stopped, _clean_tail). Реализует render()
    с собственным набором колонок, специфичным для multi-режима.
    """

    def mark_started(self, worker_id: int, long_context: bool = False) -> None:
        """Инициализирует данные воркера при старте."""
        self.workers[worker_id] = {
            "round": 1, "calls": 0, "prompt": 0, "gen": 0,
            "gen_est": 0, "chunks": 0, "call_gen": 0, "total": 0,
            "speed": 0, "avg": 0, "ttft": 0, "ttft_sum": 0,
            "wall": "", "tail": "[dim]waiting...[/]",
            "long_context": long_context,
        }

    def update_stats(self, msg: dict) -> None:
        """Расширенная версия: добавляет round, prompt, total."""
        super().update_stats(msg)
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "round": msg.get("round"),
            "prompt": msg.get("p"),
            "total": msg.get("total"),
        })

    def __rich__(self) -> Table:
        """Rich-совместимый интерфейс для Live display."""
        return self.render()

    def render(self) -> Table:
        """Рендерит таблицу для Rich Live."""
        now = datetime.now().strftime("%H:%M:%S")
        dur_str = f"{self.duration}s" if self.duration is not None else "\u221e"
        active = len([w for w in self.workers.values() if w.get("calls", 0) > 0])
        info = (
            f"Workers: {active} active / {self.total_workers} | "
            f"Duration: {dur_str} | "
            f"Model: {MODEL}"
        )

        table = Table(
            box=DOUBLE,
            show_header=True,
            title=f"[bold cyan]LLM Speed Benchmark (MULTI)[/] [dim]{now} -- {info}[/]",
            title_style="cyan",
            caption="[dim]Gen = точные токены (usage) | Gen est = оценка (Gen + чанки_в_вызове, 1 чанк ≈ 1 токен, после стрима = Gen) | Chunks = всего чанков с начала воркера | CallGen = в последнем вызове | Speed = чанки / время_вызова | Avg = Gen est / время_воркера | TTFT = последний вызов | TTFT sum = суммарный TTFT всех вызовов[/]",
            caption_style="dim",
            padding=(0, 1),
            highlight=True
        )

        # Колонки с фиксированной шириной
        table.add_column("ID", width=7, justify="right", style="bold yellow")
        table.add_column("Round", width=6, justify="right")
        table.add_column("Calls", width=6, justify="right")
        table.add_column("Prompt", width=12, justify="right")
        table.add_column("Gen", width=10, justify="right")
        table.add_column("Gen est", width=10, justify="right")
        table.add_column("Chunks", width=9, justify="right")
        table.add_column("CallGen", width=10, justify="right")
        table.add_column("Total", width=12, justify="right")
        table.add_column("Speed", width=9, justify="right")
        table.add_column("Avg", width=9, justify="right")
        table.add_column("TTFT", width=8, justify="right")
        table.add_column("TTFT sum", width=10, justify="right")
        table.add_column("Time", width=6, justify="center")
        table.add_column("Response", width=self.response_width, overflow="fold")

        # --- Строки воркеров ---
        for wid in range(self.total_workers):
            wid_str = str(wid)

            # Ошибка
            if wid in self._errors:
                table.add_row(
                    wid_str, "", "", "", "", "", "",
                    "", "", "", "[red]ERROR[/]", "",
                    "", "",
                    self._errors[wid][:55] + "..." if len(self._errors[wid]) > 55 else self._errors[wid]
                )
                continue

            # Нет в словаре -- ещё не запущен
            if wid not in self.workers:
                table.add_row(
                    wid_str, "", "", "", "", "", "",
                    "", "", "", "[dim]pend[/]", "",
                    "", "",
                    "[dim]waiting...[/]"
                )
                continue

            # Есть -- читаем из одного словаря
            w = self.workers[wid]
            tail = self._clean_tail(w.get("tail", "") or "[dim]waiting...[/]")

            # Форматирование колонок
            speed = w.get("speed", 0) or 0
            avg = w.get("avg", 0) or 0
            ttft = w.get("ttft", 0) or 0
            ttft_sum = w.get("ttft_sum", 0) or 0

            # Маркер long context в ID
            display_id = wid_str
            if w.get("long_context"):
                display_id = f"[LC]{wid_str}"

            row = [
                display_id,
                str(w.get("round", 1)) if w.get("calls", 0) > 0 else "",
                str(w["calls"]) if w.get("calls", 0) > 0 else "",
                f"{w['prompt']:>10,}" if w.get("prompt", 0) > 0 else "",
                f"{w['gen']:>9,}" if w.get("gen", 0) > 0 else "",
                f"{w['gen_est']:>9,}" if w.get("gen_est", 0) > 0 else "",
                f"{w['chunks']:>8,}" if w.get("chunks", 0) > 0 else "",
                f"{w['call_gen']:>8,}" if w.get("call_gen", 0) > 0 else "",
                f"{w['total']:>10,}" if w.get("total", 0) > 0 else "",
                f"{speed:.0f} t/s" if speed > 0 else "",
                f"{avg:.1f} t/s" if avg > 0 else "",
                f"{ttft:.2f}s" if ttft > 0 else "",
                f"{ttft_sum:.1f}s" if ttft_sum > 0 else "",
                w.get("wall", ""),
                tail,
            ]
            table.add_row(*row)

        return table


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def run_benchmark(
    workers=4,
    duration=None,
    response_width=60,
    base_url=None,
    api_key=None,
    model=None,
    max_context=None,
    long_context_workers=0,
    data_dir=None,
    split="100K",
    skip_errors=False,
):
    """Запускает многопроцессный бенчмарк.

    Args:
        workers: Количество воркеров.
        duration: Длительность в секундах (None -- до Ctrl+C).
        response_width: Ширина колонки Response.
        base_url: Переопределение BASE_URL.
        api_key: Переопределение API_KEY.
        model: Переопределение MODEL.
        max_context: Переопределение MAX_CONTEXT_TOKENS.
        long_context_workers: Количество воркеров с длинным контекстом.
        data_dir: Директория с датасетами.
        split: Сплит BEAM датасета ("100K", "500K", "1M").
        skip_errors: Если True — продолжать после ошибки.
    """
    apply_config(base_url=base_url, api_key=api_key, model=model, max_context=max_context)

    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    # Проверяем реальный контекст модели через API
    if long_context_workers > 0:
        try:
            _detect_client = _u.get_client()
            actual_context = _u.detect_model_context_length(_detect_client, _u.MODEL)
            if actual_context < _u.MAX_CONTEXT_TOKENS:
                print()
                print(f"   ОШИБКА: модель {_u.MODEL} поддерживает контекст {actual_context:,} токенов,")
                print(f"   а в настройках указано MAX_CONTEXT_TOKENS={_u.MAX_CONTEXT_TOKENS:,}.")
                print(f"   Уменьшите MAX_CONTEXT_TOKENS в .env или передайте --max-context {actual_context}.")
                print()
                sys.exit(1)
        except Exception as e:
            print(f"   Warning: не удалось определить контекст модели: {e}")

    # Подготовка long context данных
    long_context_messages = []
    if long_context_workers > 0:
        try:
            from .long_context import LongContextDataset
            ds_data_dir = data_dir or os.path.expanduser(
                "~/workspace/data/llm-speed-benchmark/datasets"
            )
            ds = LongContextDataset(data_dir=ds_data_dir)
            ds.load(split=split)
            for i in range(long_context_workers):
                conv_count = len(ds.get_conversations_info())
                msgs = ds.get_messages(
                    conversation_id=i % conv_count,
                    max_tokens=_u.MAX_CONTEXT_TOKENS - 1000,
                )
                long_context_messages.append(msgs)
            console = Console()
            console.print(
                f"   [cyan]Long context:[/] {long_context_workers} воркер(ов) "
                f"из split={split}"
            )
        except Exception as e:
            print(f"   Warning: не удалось загрузить long context: {e}")
            long_context_workers = 0
            long_context_messages = []

    dur_str = f"{duration}с" if duration is not None else "бесконечно (Ctrl+C)"
    print(f"Запуск {workers} воркеров на {dur_str}...")
    print(f"   Модель: {_u.MODEL}")
    print(f"   Ширина Response: {response_width} символов")
    print(f"   Макс контекст: {_u.MAX_CONTEXT_TOKENS:,}")
    print()

    q = Queue()
    start_event = Event()
    processes = []
    console = Console()

    def sigint_handler(signum, frame):
        console.print("\n[red]Stopped[/] Прервано пользователем. Остановка...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    live_table = LiveTable(duration, workers, response_width)
    live = Live(live_table, console=console, refresh_per_second=5)
    live.start()

    for i in range(workers):
        is_lc = i < long_context_workers
        initial_msgs = long_context_messages[i] if is_lc else None
        p = Process(
            target=worker,
            args=(i, q, start_event, duration, initial_msgs, skip_errors),
            name=f"worker-{i}",
        )
        p.start()
        processes.append(p)
        console.print(f"   Запущен воркер {i} (pid={p.pid}){' [LC]' if is_lc else ''}")

    start_event.set()
    bench_start = time.time()

    try:
        while duration is None or (time.time() - bench_start < duration + 2):
            try:
                msg = q.get(timeout=0.1)
                msg_type = msg["type"]
                msg_id = msg["id"]

                if msg_type == "start":
                    live_table.mark_started(msg_id, msg.get("long_context", False))
                elif msg_type == "stats":
                    live_table.update_stats(msg)
                elif msg_type == "live":
                    live_table.update_live(msg)
                elif msg_type == "time":
                    live_table.update_time(msg)
                elif msg_type == "error":
                    live_table.mark_error(msg_id, msg["traceback"])
                    console.print(f"[red] Воркер {msg_id} упал: {msg.get('traceback', 'unknown error')[:100]}[/]")
                elif msg_type == "error_stop":
                    live_table.mark_stopped(msg_id, msg.get("error", "unknown"))
                    console.print(f"[red] Воркер {msg_id} остановлен: {msg.get('error', 'unknown')[:100]}[/]")

                live.update(live_table)
            except Exception:
                pass
    except KeyboardInterrupt:
        sigint_handler(None, None)

    live.stop()

    # --- Итоги ---
    console.clear()

    # Ждём завершения воркеров
    for p in processes:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
            p.join()

    console.print("\n[cyan]Бенчмарк завершён.[/]")
    console.print(f"   Запущено: {len(processes)} воркеров | Длительность: {duration}с")
    console.print()

    total_avg_ttft = 0.0
    total_ttft_entries = 0
    total_gen_all = 0
    total_wall_all = 0.0
    for wid in sorted(live_table.workers.keys()):
        w = live_table.workers[wid]
        ttft = w.get("ttft", 0) or 0
        if ttft > 0:
            total_avg_ttft += ttft
            total_ttft_entries += 1
        total_gen_all += w.get("gen", 0) or 0
        # Парсим wall time "MM:SS" -> секунды
        wall_str = w.get("wall", "00:00")
        try:
            parts = wall_str.split(":")
            total_wall_all += int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            pass
        console.print(
            f"   Worker {wid}: "
            f"[cyan]{w['calls']}[/] calls | "
            f"Round: [bold]{w.get('round', 1)}[/] | "
            f"Prompt: [magenta]{w['prompt']:,}[/] | "
            f"Gen: [green]{w['gen']:,}[/] | "
            f"Avg: [yellow]{w['avg']:.1f}[/] t/s | "
            f"Time: {w.get('wall', '00:00')} | "
            f"TTFT: {w.get('ttft', 0):.2f}s"
        )

    if total_ttft_entries > 0:
        console.print()
        console.print(f"   [bold]Overall Avg TTFT: {total_avg_ttft / total_ttft_entries:.2f}s[/]")

    if total_wall_all > 0:
        overall_speed = total_gen_all / total_wall_all
        console.print(f"   [bold]Overall Avg Speed: [yellow]{overall_speed:.1f}[/] t/s "
                       f"({total_gen_all:,} tok / {format_time(total_wall_all)})")

    console.print()


def main():
    """Legacy entry point -- перенаправляет в cli()."""
    cli()


def cli():
    """CLI entry point для bench_multi."""
    parser = argparse.ArgumentParser(
        prog="bench_multi",
        description="Многопроцессный бенчмарк скорости стриминга LLM",
    )
    add_common_args(parser)
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Количество воркеров (по умолч. 4)",
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=None,
        help="Длительность теста в секундах (по умолч. None -- до Ctrl+C)",
    )
    parser.add_argument(
        "--response-width", type=int, default=60,
        help="Ширина колонки Response (по умолч. 60)",
    )
    parser.add_argument(
        "--long-context-workers", type=int, default=0,
        help="Количество воркеров с длинным контекстом из BEAM (по умолч. 0)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Директория с датасетами (по умолч. ~/workspace/data/llm-speed-benchmark/datasets)",
    )
    parser.add_argument(
        "--split", type=str, default="100K", choices=["100K", "500K", "1M"],
        help="Сплит BEAM датасета (по умолч. 100K)",
    )
    parser.add_argument(
        "--skip-errors", action="store_true", default=False,
        help="Продолжать после ошибки (по умолчанию воркер останавливается)",
    )
    args = parser.parse_args()

    run_benchmark(
        workers=args.workers,
        duration=args.duration,
        response_width=args.response_width,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_context=args.max_context,
        long_context_workers=args.long_context_workers,
        data_dir=args.data_dir,
        split=args.split,
        skip_errors=args.skip_errors,
    )


if __name__ == "__main__":
    cli()