#!/usr/bin/env python3
"""
bench_multi.py

Многопроцессный бенчмарк скорости vLLM в режиме стриминга.
Запускает N изолированных процессов, каждый независимо накапливает контекст,
выводит обновляемую таблицу с Rich Live — без скролла, на одном экране.

Использование:
  source .venv
  python bench_multi.py --workers 4 --duration 30
"""

import os
import sys
import time
import signal
import argparse
import shutil
import traceback
from datetime import datetime
from multiprocessing import Process, Queue, Event

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.box import DOUBLE, ROUNDED

# ---------------------------------------------------------------------------
# Константы и конфигурация
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("API_KEY", "sk-vllm-qwen3.5-0.8b")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "qwen3.5-0.8b")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "262144"))
TOKEN_LIMIT_WARN = int(MAX_CONTEXT_TOKENS * 0.85)

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def truncate_history(messages):
    """Удаляем самые старые пары user/assistant, оставляем system."""
    result = list(messages)
    while len(result) > 2:
        to_remove = []
        for i, m in enumerate(result):
            if m["role"] == "user" and len(to_remove) == 0:
                to_remove.append(i)
            elif m["role"] == "assistant" and len(to_remove) == 1:
                to_remove.append(i)
                break
        if not to_remove:
            break
        for i in reversed(to_remove):
            del result[i]
    return result


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Воркер
# ---------------------------------------------------------------------------

def worker(worker_id, q, start_event, duration):
    """Запускает цикл вызовов LLM в отдельном процессе."""
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)

        messages = [
            {"role": "system", "content": "Ты полезный помощник. Отвечай подробно и развёрнуто."}
        ]

        # Сразу сообщаем о старте
        q.put({"type": "start", "id": worker_id})

        start_time = time.time()
        call_count = 0
        total_gen = 0
        history_tokens = 0

        start_event.wait()  # Синхронизация старта

        while time.time() - start_time < duration:
            prompt_text = "продолжай" if call_count > 0 else "Что ты умеешь? Расскажи обо всём максимально подробно."
            messages.append({"role": "user", "content": prompt_text})

            turn_start = time.time()
            completion_tokens = 0
            assistant_content = ""

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=8192,
                    stream_options={"include_usage": True},
                )

                for chunk in response:
                    if chunk.usage:
                        history_tokens = chunk.usage.prompt_tokens or 0
                        completion_tokens = chunk.usage.completion_tokens or 0

                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta is not None:
                            content = getattr(delta, 'content', None) or ""
                            assistant_content += content
                            # Отправляем промежуточный чанк для live-обновления
                            q.put({
                                "type": "chunk",
                                "id": worker_id,
                                "tail": assistant_content[-100:]
                            })

            except Exception as e:
                assistant_content = f"[red]Error: {e}[/]"

            if assistant_content and not assistant_content.startswith("[red]Error:"):
                messages.append({"role": "assistant", "content": assistant_content})

            # Проверка и обрезка контекста
            if history_tokens + total_gen >= TOKEN_LIMIT_WARN:
                messages = truncate_history(messages)

            elapsed = time.time() - turn_start
            call_count += 1
            wall_total = time.time() - start_time

            total_gen += completion_tokens
            avg_speed = total_gen / wall_total if wall_total > 0 else 0
            curr_speed = completion_tokens / elapsed if elapsed > 0 else 0

            # Отправляем финальную статистику
            q.put({
                "type": "stats",
                "id": worker_id,
                "calls": call_count,
                "p": history_tokens,
                "g": total_gen,
                "cg": completion_tokens,
                "total": history_tokens + completion_tokens,
                "last_speed": round(curr_speed, 1),
                "avg_speed": round(avg_speed, 1),
                "tail": assistant_content[-80:] if assistant_content else "",
                "wall": format_time(wall_total)
            })

    except Exception as e:
        # Ловим ошибки на уровне воркера (не может импортировать, не может создать клиент и т.д.)
        q.put({
            "type": "error",
            "id": worker_id,
            "traceback": traceback.format_exc()
        })


# ---------------------------------------------------------------------------
# Rich Live Table
# ---------------------------------------------------------------------------

class LiveTable:
    """Динамическая таблица для Rich Live."""

    def __init__(self, duration, total_workers, response_width=60):
        self.duration = duration
        self.total_workers = total_workers
        self.response_width = response_width
        self.stats = {}          # {id: stat_dict}
        self.live_responses = {}  # {id: tail_string}
        self.console = Console()
        self._last_time = ""
        self._started = set()
        self._errors = {}  # {id: traceback}

    def update_stats(self, stats):
        self.stats[stats["id"]] = stats

    def update_response(self, worker_id, tail):
        self.live_responses[worker_id] = tail

    def mark_started(self, worker_id):
        self._started.add(worker_id)

    def mark_error(self, worker_id, traceback_str):
        self._errors[worker_id] = traceback_str

    def __rich__(self):
        """Рендерит таблицу для Rich Live."""
        now = datetime.now().strftime("%H:%M:%S")
        info = (
            f"Workers: {len(self.stats)} active / {self.total_workers} | "
            f"Duration: {self.duration}s | "
            f"Model: {MODEL}"
        )
        table = Table(
            box=DOUBLE,
            show_header=True,
            title=f"[bold cyan]LLM Speed Benchmark (MULTI)[/] [dim]{now} — {info}[/]",
            title_style="cyan",
            padding=(0, 1),
            highlight=True
        )

        # Колонки с фиксированной шириной
        table.add_column("ID", width=4, justify="right", style="bold yellow")
        table.add_column("Calls", width=6, justify="right")
        table.add_column("Prompt", width=12, justify="right")
        table.add_column("Gen cum", width=11, justify="right")
        table.add_column("CallGen", width=10, justify="right")
        table.add_column("Total", width=12, justify="right")
        table.add_column("Speed", width=14, justify="right")
        table.add_column("Time", width=6, justify="center")
        table.add_column("Response", width=self.response_width, overflow="fold")

        # --- Строки воркеров ---
        active_ids = set(self.stats.keys()) | self._started
        for wid in range(max(self.total_workers, len(active_ids))):
            wid_str = str(wid)

            # Проверяем ошибки
            if wid in self._errors:
                table.add_row(
                    wid_str,
                    "", "", "", "", "",
                    "[red]ERROR[/]",
                    "",
                    self._errors[wid][:55] + "..." if len(self._errors[wid]) > 55 else self._errors[wid]
                )
                continue

            # Проверяем, запущен ли
            if wid not in active_ids:
                table.add_row(
                    wid_str,
                    "", "", "", "", "",
                    "[dim]pending[/]",
                    "",
                    "[dim]waiting...[/]"
                )
                continue

            if wid not in self.stats:
                # Запущен, но ещё нет статистики
                tail = self.live_responses.get(wid, "[dim]waiting...[/]")
                table.add_row(
                    wid_str,
                    "", "", "", "", "",
                    "[dim]running[/]",
                    "",
                    tail
                )
                continue

            s = self.stats[wid]

            # Response: берём live-хвост если есть, иначе финальный
            tail = self.live_responses.get(wid, s.get("tail", "") or "")
            # Убираем ВСЕ управляющие символы (newline, tab, null, control chars)
            tail = "".join(c if c.isprintable() else " " for c in tail)
            if len(tail) > self.response_width - 3:
                tail = tail[:self.response_width - 3] + "..."

            call_gen = s.get("cg", 0) or 0

            row = [
                wid_str,
                str(s["calls"]),
                f"{s['p']:>10,}",
                f"{s['g']:>9,}",
                f"{call_gen:>8,}",
                f"{s['total']:>10,}",
                f"{s['avg_speed']:.1f} tok/s",
                s["wall"],
                tail
            ]
            table.add_row(*row)

        return table


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Многопроцессный бенчмарк vLLM")
    parser.add_argument("--workers", type=int, default=4, help="Количество воркеров (по умолч. 4)")
    parser.add_argument("--duration", type=int, default=30, help="Длительность теста в секундах (по умолч. 30)")
    parser.add_argument("--response-width", type=int, default=60, help="Ширина колонки Response в символах (по умолч. 60)")
    args = parser.parse_args()

    workers = args.workers
    duration = args.duration
    response_width = args.response_width

    print(f"🚀 Запуск {workers} воркеров на {duration} секунд...")
    print(f"   Модель: {MODEL}")
    print(f"   Ширина Response: {response_width} символов")
    print(f"   Макс контекст: {MAX_CONTEXT_TOKENS:,}")
    print()

    q = Queue()
    start_event = Event()
    processes = []

    def sigint_handler(signum, frame):
        console.print("\n[red]⏹[/] Прервано пользователем. Остановка...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    console = Console()
    signal.signal(signal.SIGINT, sigint_handler)

    live_table = LiveTable(duration, workers, response_width)

    with Live(live_table, console=console, refresh_per_second=5, screen=True) as live:
        for i in range(workers):
            p = Process(target=worker, args=(i, q, start_event, duration), name=f"worker-{i}")
            p.start()
            processes.append(p)
            console.print(f"   Запущен воркер {i} (pid={p.pid})")

        start_event.set()
        bench_start = time.time()

        try:
            while time.time() - bench_start < duration + 2:
                try:
                    msg = q.get(timeout=0.1)
                    msg_type = msg["type"]
                    msg_id = msg["id"]

                    if msg_type == "start":
                        live_table.mark_started(msg_id)
                    elif msg_type == "stats":
                        live_table.update_stats(msg)
                    elif msg_type == "chunk":
                        live_table.update_response(msg_id, msg["tail"])
                    elif msg_type == "error":
                        live_table.mark_error(msg_id, msg["traceback"])
                        console.print(f"[red] Воркер {msg_id} упал: {msg.get('traceback', 'unknown error')[:100]}[/]")
                except Exception:
                    pass
        except KeyboardInterrupt:
            sigint_handler(None, None)

    # --- Итоги ---
    console.clear()

    dead_workers = []
    for i, p in enumerate(processes):
        if not p.is_alive():
            dead_workers.append(i)
        else:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join()

    console.print("\n[cyan]✅ Бенчмарк завершён.[/]")
    console.print(f"   Запущено: {len(processes)} | Упало: {len(dead_workers)}")

    if dead_workers:
        console.print(f"   [red]⚠️ Упавшие воркеры:[/] {dead_workers}")

    console.print()
    for wid in sorted(live_table.stats.keys()):
        s = live_table.stats[wid]
        console.print(
            f"   Worker {wid}: "
            f"[cyan]{s['calls']}[/] calls | "
            f"Prompt: [magenta]{s['p']:,}[/] | "
            f"Gen: [green]{s['g']:,}[/] | "
            f"Avg: [yellow]{s['avg_speed']:.1f}[/] tok/s"
        )


if __name__ == "__main__":
    main()
