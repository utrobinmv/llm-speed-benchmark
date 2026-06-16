#!/usr/bin/env python3
"""
llm_speed_benchmark/bench_multi.py

Многопроцессный бенчмарк скорости vLLM в режиме стриминга.
Запускает N изолированных процессов, каждый независимо накапливает контекст,
выводит обновляемую таблицу с Rich Live — без скролла, на одном экране.

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
    MODEL,
    MAX_CONTEXT_TOKENS,
    INSTANT_WINDOW,
    _token_limit_warn,
)


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
        round_num = 1
        total_ttft = 0.0
        ttft_count = 0

        start_event.wait()  # Синхронизация старта

        while duration is None or (time.time() - start_time < duration):
            # Уникальные промпты с ID воркера и раундом
            if call_count == 0:
                prompt_text = f"Поток {worker_id:07d}. Что ты умеешь? Расскажи обо всём максимально подробно."
            else:
                prompt_text = f"Поток {worker_id:07d} (раунд {round_num}) продолжай"

            # Проверка лимита контекста — новый раунд
            if history_tokens + total_gen >= MAX_CONTEXT_TOKENS:
                messages = [{"role": "system", "content": "Ты полезный помощник. Отвечай подробно и развёрнуто."}]
                round_num += 1
                call_count = 0
                total_gen = 0
                history_tokens = 0
                total_ttft = 0.0
                ttft_count = 0

            if history_tokens + total_gen >= _token_limit_warn():
                messages = truncate_history(messages)

            messages.append({"role": "user", "content": prompt_text})

            turn_start = time.time()
            completion_tokens = 0
            assistant_content = ""
            chunk_count = 0
            first_token_time = None
            instant_buffer = []
            last_chunk_send = 0  # throttle для chunk-сообщений

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
                            if content:
                                assistant_content += content
                                chunk_count += 1
                                now_chunk = time.time()
                                if first_token_time is None:
                                    first_token_time = now_chunk
                                instant_buffer.append((now_chunk, chunk_count))

                                # Throttle: отправляем live-обновления не чаще 2 раз/сек
                                if now_chunk - last_chunk_send >= 0.5:
                                    # Мгновенная скорость
                                    inst_sp = 0
                                    if len(instant_buffer) > INSTANT_WINDOW:
                                        ws = instant_buffer[-INSTANT_WINDOW][0]
                                        we = instant_buffer[-1][0]
                                        wt = instant_buffer[-1][1] - instant_buffer[-INSTANT_WINDOW - 1][1] if len(instant_buffer) > INSTANT_WINDOW + 1 else INSTANT_WINDOW
                                        wtime = we - ws
                                        inst_sp = round(wt / wtime, 1) if wtime > 0 else 0

                                    # TTFT
                                    ttft = round(first_token_time - turn_start, 2) if first_token_time is not None else 0

                                    # Средняя скорость за всё время воркера
                                    wall_elapsed = time.time() - start_time
                                    avg_sp = round(chunk_count / wall_elapsed, 1) if wall_elapsed > 0 else 0

                                    q.put({
                                        "type": "live",
                                        "id": worker_id,
                                        "tail": assistant_content[-100:],
                                        "tok": chunk_count,
                                        "inst": inst_sp,
                                        "avg": avg_sp,
                                        "ttft": ttft,
                                    })
                                    last_chunk_send = now_chunk

            except Exception as e:
                assistant_content = f"[red]Error: {e}[/]"

            if assistant_content and not assistant_content.startswith("[red]Error:"):
                messages.append({"role": "assistant", "content": assistant_content})

            elapsed = time.time() - turn_start
            call_count += 1
            wall_total = time.time() - start_time

            # Fallback: если usage не пришло — используем chunk_count
            if completion_tokens == 0:
                completion_tokens = chunk_count

            # Защита: пустой ответ
            if completion_tokens == 0:
                q.put({
                    "type": "stats",
                    "id": worker_id,
                    "calls": call_count,
                    "p": history_tokens,
                    "g": total_gen,
                    "cg": 0,
                    "total": history_tokens,
                    "speed": 0,
                    "avg_speed": 0,
                    "inst_speed": 0,
                    "ttft": 0,
                    "tail": "⚠️ empty",
                    "wall": format_time(wall_total),
                    "round": round_num,
                })
                continue

            # TTFT
            if first_token_time is not None:
                turn_ttft = first_token_time - turn_start
                total_ttft += turn_ttft
                ttft_count += 1
            else:
                turn_ttft = 0

            total_gen += completion_tokens
            avg_speed = total_gen / wall_total if wall_total > 0 else 0
            curr_speed = completion_tokens / elapsed if elapsed > 0 else 0

            # Мгновенная скорость
            inst_speed = 0
            if len(instant_buffer) > INSTANT_WINDOW:
                window_start = instant_buffer[-INSTANT_WINDOW][0]
                window_end = instant_buffer[-1][0]
                window_tokens = instant_buffer[-1][1] - instant_buffer[-INSTANT_WINDOW - 1][1] if len(instant_buffer) > INSTANT_WINDOW + 1 else INSTANT_WINDOW
                window_time = window_end - window_start
                inst_speed = window_tokens / window_time if window_time > 0 else 0

            # Отправляем финальную статистику
            q.put({
                "type": "stats",
                "id": worker_id,
                "calls": call_count,
                "p": history_tokens,
                "g": total_gen,
                "cg": completion_tokens,
                "total": history_tokens + completion_tokens,
                "speed": round(curr_speed, 1),
                "avg_speed": round(avg_speed, 1),
                "inst_speed": round(inst_speed, 1),
                "ttft": round(turn_ttft, 2),
                "tail": assistant_content[-80:] if assistant_content else "",
                "wall": format_time(wall_total),
                "round": round_num,
            })

    except Exception as e:
        # Ловим ошибки на уровне воркера
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
        self.live_metrics = {}    # {id: {tok, inst, ttft}}
        self.console = Console()
        self._last_time = ""
        self._started = set()
        self._errors = {}  # {id: traceback}

    def update_stats(self, stats):
        self.stats[stats["id"]] = stats

    def update_response(self, worker_id, tail):
        self.live_responses[worker_id] = tail

    def update_live(self, worker_id, metrics):
        self.live_metrics[worker_id] = metrics
        # Также обновляем tail
        if "tail" in metrics:
            self.live_responses[worker_id] = metrics["tail"]

    def _clean_tail(self, tail):
        """Очищает и обрезает текст ответа для колонки Response.

        Заменяет wide-символы (CJK, эмодзи, пиктограммы — 2+ ячейки) на точки,
        так как они ломают ширину колонки. Кириллица, латиница и другие
        1-ячеечные символы сохраняются.
        """
        if not tail:
            return tail

        # Проверяем, является ли символ wide (2+ ячейки в терминале)
        def is_wide(c):
            cp = ord(c)
            # CJK Unified Ideographs и расширения
            if 0x4E00 <= cp <= 0x9FFF:
                return True
            if 0x3400 <= cp <= 0x4DBF:
                return True
            if 0x20000 <= cp <= 0x2A6DF:
                return True
            if 0x2A700 <= cp <= 0x2B73F:
                return True
            if 0x2B740 <= cp <= 0x2B81F:
                return True
            if 0x2B820 <= cp <= 0x2CEAF:
                return True
            if 0xF900 <= cp <= 0xFAFF:
                return True
            if 0x2F800 <= cp <= 0x2FA1F:
                return True
            # CJK symbols and punctuation
            if 0x3000 <= cp <= 0x303F:
                return True
            if 0xFF01 <= cp <= 0xFF60:
                return True
            # Hangul
            if 0xAC00 <= cp <= 0xD7AF:
                return True
            if 0xD7B0 <= cp <= 0xD7FF:
                return True
            # Katakana/Hiragana (некоторые wide)
            if 0x30A0 <= cp <= 0x30FF:
                return True
            if 0x3040 <= cp <= 0x309F:
                return True
            # Бамбу/Тай/Гуарани и другие SE-Asian
            if 0x0E01 <= cp <= 0x0E3E:
                return True
            if 0x0E40 <= cp <= 0x0E4E:
                return True
            # Эмодзи и другие supplementary symbols (2 ячейки)
            if cp >= 0x1F000:
                return True
            if 0x1F300 <= cp <= 0x1F9FF:
                return True
            if 0x1FA00 <= cp <= 0x1FA6F:
                return True
            if 0x1FA70 <= cp <= 0x1FAFF:
                return True
            if 0x2600 <= cp <= 0x26FF:  # Misc Symbols
                return True
            if 0x2700 <= cp <= 0x27BF:  # Dingbats
                return True
            if 0x2300 <= cp <= 0x23FF:  # Misc Technical (⏳⌚⏱⏲⏰▶⏸⏹⏺⏏⏪⏫⏬)
                return True
            if 0xFE00 <= cp <= 0xFE0F:  # Variation selectors — skip
                return False
            if 0xFE30 <= cp <= 0xFE4F:
                return True
            if 0x2000 <= cp <= 0x206F:  # General punctuation — narrow
                return False
            return False

        # --- Проходим по строке, заменяя wide символы ---
        clean = ""
        in_tag = False
        for c in tail:
            if c == '[':
                in_tag = True
                clean += c
                continue
            if c == ']' and in_tag:
                in_tag = False
                clean += c
                continue
            if in_tag:
                clean += c
                continue
            # Вне тега: заменяем проблемные символы
            if c == '\n' or c == '\r' or c == '\t':
                clean += " "
            elif c.isprintable() and not is_wide(c):
                clean += c
            else:
                clean += "."

        # Схлопываем множественные пробелы (только вне тегов)
        prev = None
        while prev != clean:
            prev = clean
            result = ""
            in_tag = False
            prev_space = False
            for c in prev:
                if c == '[':
                    in_tag = True
                    result += c
                    prev_space = False
                    continue
                if c == ']' and in_tag:
                    in_tag = False
                    result += c
                    prev_space = False
                    continue
                if in_tag:
                    result += c
                    prev_space = False
                    continue
                if c == ' ' and prev_space:
                    continue
                prev_space = (c == ' ')
                result += c
            clean = result

        clean = clean.strip()
        if not clean:
            return tail

        # --- Обрезаем до ширины колонки (считаем только видимые символы) ---
        def visual_length(s):
            length = 0
            in_t = False
            for ch in s:
                if ch == '[':
                    in_t = True
                    continue
                if ch == ']' and in_t:
                    in_t = False
                    continue
                if not in_t:
                    length += 1
            return length

        vlen = visual_length(clean)
        max_vlen = self.response_width - 3  # место под "..."
        if vlen > max_vlen:
            truncated = ""
            count = 0
            in_t = False
            for ch in clean:
                if ch == '[':
                    in_t = True
                    truncated += ch
                    continue
                if ch == ']' and in_t:
                    in_t = False
                    truncated += ch
                    continue
                if in_t:
                    truncated += ch
                    continue
                if count >= max_vlen:
                    break
                count += 1
                truncated += ch
            clean = "..." + truncated

        return clean

    def mark_started(self, worker_id):
        self._started.add(worker_id)

    def mark_error(self, worker_id, traceback_str):
        self._errors[worker_id] = traceback_str

    def __rich__(self):
        """Рендерит таблицу для Rich Live."""
        now = datetime.now().strftime("%H:%M:%S")
        dur_str = f"{self.duration}s" if self.duration is not None else "∞"
        info = (
            f"Workers: {len(self.stats)} active / {self.total_workers} | "
            f"Duration: {dur_str} | "
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
        table.add_column("Round", width=6, justify="right")
        table.add_column("Calls", width=6, justify="right")
        table.add_column("Prompt", width=12, justify="right")
        table.add_column("Gen cum", width=11, justify="right")
        table.add_column("CallGen", width=10, justify="right")
        table.add_column("Total", width=12, justify="right")
        table.add_column("Speed", width=9, justify="right")
        table.add_column("Avg", width=9, justify="right")
        table.add_column("TTFT", width=8, justify="right")
        table.add_column("Time", width=6, justify="center")
        table.add_column("Response", width=self.response_width, overflow="fold")

        # --- Строки воркеров ---
        active_ids = set(self.stats.keys()) | self._started
        for wid in range(max(self.total_workers, len(active_ids))):
            wid_str = str(wid)

            # Проверяем ошибки
            if wid in self._errors:
                table.add_row(
                    wid_str, "", "", "", "", "", "",
                    "", "[red]ERROR[/]", "",
                    "",
                    self._errors[wid][:55] + "..." if len(self._errors[wid]) > 55 else self._errors[wid]
                )
                continue

            # Проверяем, запущен ли
            if wid not in active_ids:
                table.add_row(
                    wid_str, "", "", "", "", "", "",
                    "", "[dim]pend[/]", "",
                    "",
                    "[dim]waiting...[/]"
                )
                continue

            if wid not in self.stats:
                # Запущен, но ещё нет финальной статистики — показываем live-метрики
                lm = self.live_metrics.get(wid, {})
                tail = self._clean_tail(self.live_responses.get(wid, "[dim]waiting...[/]"))
                inst = lm.get("inst", 0) or 0
                avg = lm.get("avg", 0) or 0
                ttft = lm.get("ttft", 0) or 0
                speed_str = f"{inst:.0f} t/s" if inst > 0 else ""
                avg_str = f"{avg:.1f} t/s" if avg > 0 else ""
                ttft_str = f"{ttft:.2f}s" if ttft > 0 else ""
                table.add_row(
                    wid_str, "", "", "", "", "", "",
                    speed_str, avg_str, ttft_str,
                    "",
                    tail
                )
                continue

            s = self.stats[wid]

            # Response: берём live-хвост если есть, иначе финальный
            tail = self._clean_tail(self.live_responses.get(wid, s.get("tail", "") or ""))

            call_gen = s.get("cg", 0) or 0

            # Speed = мгновенная, Avg = средняя за всё время
            lm = self.live_metrics.get(wid, {})
            avg_sp = s.get("avg_speed", 0) or 0
            avg_str = f"{avg_sp:.1f} t/s" if avg_sp > 0 else ""
            if lm:
                inst = lm.get("inst", 0) or 0
                speed_str = f"{inst:.0f} t/s" if inst > 0 else ""
                ttft = lm.get("ttft", 0) or 0
            else:
                inst_sp = s.get("inst_speed", 0) or 0
                speed_str = f"{inst_sp:.0f} t/s" if inst_sp > 0 else ""
                ttft = s.get("ttft", 0) or 0
            ttft_str = f"{ttft:.2f}s" if ttft > 0 else ""

            row = [
                wid_str,
                str(s.get("round", 1)),
                str(s["calls"]),
                f"{s['p']:>10,}",
                f"{s['g']:>9,}",
                f"{call_gen:>8,}",
                f"{s['total']:>10,}",
                speed_str,
                avg_str,
                ttft_str,
                s["wall"],
                tail
            ]
            table.add_row(*row)

        return table


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def run_benchmark(workers=4, duration=None, response_width=60, base_url=None, api_key=None, model=None, max_context=None):
    """Запускает многопроцессный бенчмарк.

    Args:
        workers: Количество воркеров.
        duration: Длительность в секундах (None — до Ctrl+C).
        response_width: Ширина колонки Response.
        base_url: Переопределение BASE_URL.
        api_key: Переопределение API_KEY.
        model: Переопределение MODEL.
        max_context: Переопределение MAX_CONTEXT_TOKENS.
    """
    # Приоритет: CLI > .env
    import llm_speed_benchmark.utils as _u
    if base_url is not None:
        _u.BASE_URL = base_url
    if api_key is not None:
        _u.API_KEY = api_key
    if model is not None:
        _u.MODEL = model
    if max_context is not None:
        _u.MAX_CONTEXT_TOKENS = max_context
        _u.TOKEN_LIMIT_WARN = _token_limit_warn()

    dur_str = f"{duration}с" if duration is not None else "бесконечно (Ctrl+C)"
    print(f"🚀 Запуск {workers} воркеров на {dur_str}...")
    print(f"   Модель: {_u.MODEL}")
    print(f"   Ширина Response: {response_width} символов")
    print(f"   Макс контекст: {_u.MAX_CONTEXT_TOKENS:,}")
    print(f"   Instant window: {INSTANT_WINDOW} tok")
    print()

    q = Queue()
    start_event = Event()
    processes = []
    console = Console()

    def sigint_handler(signum, frame):
        console.print("\n[red]⏹[/] Прервано пользователем. Остановка...")
        for p in processes:
            p.terminate()
        sys.exit(0)

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
            while duration is None or (time.time() - bench_start < duration + 2):
                try:
                    msg = q.get(timeout=0.1)
                    msg_type = msg["type"]
                    msg_id = msg["id"]

                    if msg_type == "start":
                        live_table.mark_started(msg_id)
                    elif msg_type == "stats":
                        live_table.update_stats(msg)
                    elif msg_type == "live":
                        live_table.update_live(msg_id, msg)
                    elif msg_type == "error":
                        live_table.mark_error(msg_id, msg["traceback"])
                        console.print(f"[red] Воркер {msg_id} упал: {msg.get('traceback', 'unknown error')[:100]}[/]")
                except Exception:
                    pass
        except KeyboardInterrupt:
            sigint_handler(None, None)

    # --- Итоги ---
    console.clear()

    # Ждём завершения воркеров
    for p in processes:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
            p.join()

    console.print("\n[cyan]✅ Бенчмарк завершён.[/]")
    console.print(f"   Запущено: {len(processes)} воркеров | Длительность: {duration}с")
    console.print()

    total_avg_ttft = 0.0
    total_ttft_entries = 0
    total_gen_all = 0
    total_wall_all = 0.0
    for wid in sorted(live_table.stats.keys()):
        s = live_table.stats[wid]
        ttft = s.get("ttft", 0) or 0
        if ttft > 0:
            total_avg_ttft += ttft
            total_ttft_entries += 1
        total_gen_all += s.get("g", 0) or 0
        # Парсим wall time "MM:SS" → секунды
        wall_str = s.get("wall", "00:00")
        try:
            parts = wall_str.split(":")
            total_wall_all += int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            pass
        console.print(
            f"   Worker {wid}: "
            f"[cyan]{s['calls']}[/] calls | "
            f"Round: [bold]{s.get('round', 1)}[/] | "
            f"Prompt: [magenta]{s['p']:,}[/] | "
            f"Gen: [green]{s['g']:,}[/] | "
            f"Avg: [yellow]{s['avg_speed']:.1f}[/] t/s | "
            f"Time: {s.get('wall', '00:00')} | "
            f"TTFT: {s.get('ttft', 0):.2f}s"
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
    """Legacy entry point — перенаправляет в cli()."""
    cli()


def cli():
    """CLI entry point для bench_multi."""
    parser = argparse.ArgumentParser(
        prog="bench_multi",
        description="Многопроцессный бенчмарк скорости стриминга LLM",
    )
    parser.add_argument("--base-url", "-u", type=str, default=None, help="Адрес API (OpenAI-compatible)")
    parser.add_argument("--api-key", "-k", type=str, default=None, help="API ключ")
    parser.add_argument("--model", "-m", type=str, default=None, help="Название модели")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Количество воркеров (по умолч. 4)")
    parser.add_argument("--duration", "-d", type=int, default=None, help="Длительность теста в секундах (по умолч. None — до Ctrl+C)")
    parser.add_argument("--response-width", type=int, default=60, help="Ширина колонки Response (по умолч. 60)")
    parser.add_argument("--max-context", type=int, default=None, help="Переопределение MAX_CONTEXT_TOKENS")
    args = parser.parse_args()

    run_benchmark(
        workers=args.workers,
        duration=args.duration,
        response_width=args.response_width,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_context=args.max_context,
    )


if __name__ == "__main__":
    cli()
