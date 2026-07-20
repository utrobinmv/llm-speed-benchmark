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


def worker(  # type: ignore[reportInvalidTypeForm]
    worker_id: int,
    q: "Queue",
    start_event: "Event",
    duration: int | None,
    initial_messages: list[dict] | None = None,
) -> None:
    """Запускает цикл вызовов LLM в отдельном процессе.

    Args:
        worker_id: ID воркера.
        q: Queue для сообщений.
        start_event: Event для синхронизации старта.
        duration: Длительность в секундах.
        initial_messages: Начальные сообщения для long context (из датасета).
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
        history_tokens = 0
        round_num = 1
        total_ttft = 0.0
        ttft_count = 0

        start_event.wait()  # Синхронизация старта

        # --- Shared state для потока time_sender ---
        from threading import Lock
        _state_lock = Lock()
        _state = {
            'total_gen': 0,
            'tokens_per_chunk': 1.0,
            'chunk_count': 0,
            'total_ttft': 0.0,
        }

        # --- Поток для обновления wall-time каждую секунду ---
        def _time_sender():
            while True:
                time.sleep(1.0)
                wall = time.time() - start_time
                if wall > 0:
                    with _state_lock:
                        tg = _state['total_gen']
                        tpc = _state['tokens_per_chunk']
                        cc = _state['chunk_count']
                    # Пересчитываем Avg с актуальным временем
                    est_gen = tg + (cc * tpc)
                    avg_sp = round(est_gen / wall, 1) if wall > 0 else 0
                    q.put({
                        "type": "time",
                        "id": worker_id,
                        "wall": format_time(wall),
                        "avg": avg_sp,
                    })

        from threading import Thread
        _time_thread = Thread(target=_time_sender, daemon=True)
        _time_thread.start()

        # --- Callback для live-обновлений во время стриминга ---
        def _on_chunk(chunk_count, inst_sp, avg_sp, ttft, tail, wall_elapsed):
            # TTFT sum = накопленный до этого вызова + TTFT текущего (если уже есть)
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
            })

        while duration is None or (time.time() - start_time < duration):
            # Уникальные промпты с ID воркера и раундом
            if call_count == 0:
                prompt_text = f"Поток {worker_id:07d}. Что ты умеешь? Расскажи обо всём максимально подробно."
            else:
                prompt_text = f"Поток {worker_id:07d} (раунд {round_num}) продолжай"

            # Проверка лимита контекста -- новый раунд
            if history_tokens + total_gen >= MAX_CONTEXT_TOKENS:
                messages = [{"role": "system", "content": "Ты полезный помощник. Отвечай подробно и развёрнуто."}]
                round_num += 1
                call_count = 0
                total_gen = 0
                # total_chunks НЕ сбрасываем -- кумулятивный счётчик с момента старта
                history_tokens = 0
                total_ttft = 0.0
                ttft_count = 0

            if history_tokens + total_gen >= _token_limit_warn():
                messages = truncate_history(messages)

            messages.append({"role": "user", "content": prompt_text})

            turn_start = time.time()
            assistant_content = ""
            metrics = None

            try:
                # Настройка callback для live-обновлений
                session.on_chunk = _on_chunk
                session.on_chunk_args = {
                    "total_gen": total_gen,
                    "start_time": start_time,
                }
                session.call_count = call_count

                # === Streaming через StreamSession ===
                metrics = session.run(
                    messages=messages,
                    model=MODEL,
                )

                assistant_content = metrics.assistant_content

            except Exception as e:
                assistant_content = f"[red]Error: {e}[/]"

            if assistant_content and not assistant_content.startswith("[red]Error:"):
                messages.append({"role": "assistant", "content": assistant_content})

            elapsed = time.time() - turn_start
            call_count += 1
            wall_total = time.time() - start_time

            # Если ошибка в stream -- пропускаем
            if metrics is None:
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
                    "tail": assistant_content[:80] if assistant_content else "error",
                    "wall": format_time(wall_total),
                    "round": round_num,
                })
                continue

            completion_tokens = metrics.completion_tokens
            chunk_count = metrics.chunk_count

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
                    "tail": "empty",
                    "wall": format_time(wall_total),
                    "round": round_num,
                })
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

            # Обновляем shared state для потока time_sender
            with _state_lock:
                _state['total_gen'] = total_gen
                _state['tokens_per_chunk'] = session.tokens_per_chunk
                _state['total_ttft'] = total_ttft
                _state['chunk_count'] = 0  # сброс для нового вызова

            avg_speed = total_gen / wall_total if wall_total > 0 else 0

            # Отправляем финальную статистику
            q.put({
                "type": "stats",
                "id": worker_id,
                "calls": call_count,
                "p": history_tokens,
                "g": total_gen,
                "cg": completion_tokens,
                "chunks": total_chunks,
                "est_gen": total_gen,  # после стрима = точное значение
                "total": history_tokens + completion_tokens,
                "speed": round(metrics.call_speed, 1),
                "avg_speed": round(avg_speed, 1),
                "inst_speed": round(metrics.instant_speed, 1),
                "ttft": round(turn_ttft, 2),
                "ttft_sum": round(total_ttft, 2),
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
    """Живая таблица для Rich Live display.

    ОДИН словарь на воркер -- все обновления пишут в него, рендер читает из него.
    """

    def __init__(self, duration, total_workers, response_width=60):
        self.duration = duration
        self.total_workers = total_workers
        self.response_width = response_width
        self.workers = {}     # {id: {round, calls, prompt, gen, gen_est, chunks,
                              #        call_gen, total, speed, avg, ttft, ttft_sum,
                              #        wall, tail}}
        self._errors = {}     # {id: traceback}
        self.console = Console()

    @staticmethod
    def _merge(w, data):
        """Merge non-None values from data into worker dict w."""
        for k, v in data.items():
            if v is not None:
                w[k] = v

    def mark_started(self, worker_id, long_context=False):
        self.workers[worker_id] = {
            "round": 1, "calls": 0, "prompt": 0, "gen": 0,
            "gen_est": 0, "chunks": 0, "call_gen": 0, "total": 0,
            "speed": 0, "avg": 0, "ttft": 0, "ttft_sum": 0,
            "wall": "", "tail": "[dim]waiting...[/]",
            "long_context": long_context,
        }

    def update_stats(self, msg):
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "round": msg.get("round"),
            "calls": msg.get("calls"),
            "prompt": msg.get("p"),
            "gen": msg.get("g"),
            "gen_est": msg.get("est_gen"),
            "chunks": msg.get("chunks"),
            "call_gen": msg.get("cg"),
            "total": msg.get("total"),
            "speed": msg.get("inst_speed"),
            "avg": msg.get("avg_speed"),
            "ttft": msg.get("ttft"),
            "ttft_sum": msg.get("ttft_sum"),
            "wall": msg.get("wall"),
            "tail": msg.get("tail"),
        })

    def update_live(self, msg):
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
        })

    def update_time(self, msg):
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "wall": msg.get("wall"),
            "avg": msg.get("avg"),
        })

    def mark_error(self, worker_id, traceback_str):
        self._errors[worker_id] = traceback_str

    def _clean_tail(self, tail):
        """Очищает и обрезает текст ответа для колонки Response.

        Заменяет wide-символы (CJK, эмодзи, пиктограммы -- 2+ ячейки) на точки,
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
            if 0x2300 <= cp <= 0x23FF:  # Misc Technical
                return True
            if 0xFE00 <= cp <= 0xFE0F:  # Variation selectors -- skip
                return False
            if 0xFE30 <= cp <= 0xFE4F:
                return True
            if 0x2000 <= cp <= 0x206F:  # General punctuation -- narrow
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

    def __rich__(self):
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
            caption="[dim]Gen = точные токены (usage) | Gen est = оценка (Gen + чанки_в_вызове x tokens_per_chunk, после стрима = Gen) | Chunks = всего чанков с начала воркера | CallGen = в последнем вызове | Speed = скорость последних 5 чанков | Avg = Gen est / время_воркера (обновляется каждую сек) | TTFT = последний вызов | TTFT sum = суммарный TTFT всех вызовов[/]",
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

        # --- Строки воркеров -- ОДНА ветка ---
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

    with Live(live_table, console=console, refresh_per_second=5, screen=True) as live:
        for i in range(workers):
            is_lc = i < long_context_workers
            initial_msgs = long_context_messages[i] if is_lc else None
            p = Process(
                target=worker,
                args=(i, q, start_event, duration, initial_msgs),
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
    )


if __name__ == "__main__":
    cli()
