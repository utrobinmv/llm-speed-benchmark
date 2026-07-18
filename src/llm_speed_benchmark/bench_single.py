#!/usr/bin/env python3
"""
llm_speed_benchmark/bench_single.py

Бенчмарк скорости vLLM в режиме стриминга. Не выводит текст ответа,
только статистику. Работает до заполнения контекстного окна.

Использование:
  bench_single
  bench_single --duration 60
"""

import sys
import time
import argparse

from .utils import (
    get_client,
    truncate_history,
    progress_bar,
    format_time,
    _token_limit_warn,
)
from .cli_common import add_common_args, apply_config
from .streaming import StreamSession


def run_benchmark(duration=None, base_url=None, api_key=None, model=None, max_context=None):
    """Запускает последовательный бенчмарк.

    Args:
        duration: Ограничение по времени в секундах (None -- до лимита контекста).
        base_url: Переопределение BASE_URL.
        api_key: Переопределение API_KEY.
        model: Переопределение MODEL.
        max_context: Переопределение MAX_CONTEXT_TOKENS.
    """
    apply_config(base_url=base_url, api_key=api_key, model=model, max_context=max_context)

    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    client = get_client()
    session = StreamSession(client)

    messages = [
        {
            "role": "system",
            "content": "Ты полезный помощник. Отвечай подробно и развёрнуто.",
        }
    ]

    prompts = [
        "Что ты умеешь? Расскажи обо всём максимально подробно.",
        "продолжай",
    ]

    print("=" * 70)
    print("  LLM Speed Benchmark (текст скрыт)")
    print("=" * 70)
    print(f"  Модель:         {_u.MODEL}")
    print(f"  Max context:    {_u.MAX_CONTEXT_TOKENS:,}")
    print(f"  Предупреждение: {_token_limit_warn():,} (85%)")
    if duration is not None:
        print(f"  Длительность:   {duration}с")
    print("=" * 70)
    print()

    total_completion_tokens = 0
    history_tokens = 0
    start_wall = time.time()
    turn = 0
    last_stats_time = time.time()

    # Суммарный TTFT (Time To First Token) по всем turn'ам
    total_ttft = 0.0
    ttft_count = 0

    try:
        while True:
            # Проверка по времени (duration is not None -- корректно для 0)
            if duration is not None and (time.time() - start_wall) >= duration:
                break

            turn += 1
            prompt_text = prompts[0 if turn == 1 else 1]

            # Проверка контекста
            if history_tokens + total_completion_tokens >= _u.MAX_CONTEXT_TOKENS:
                break

            if history_tokens + total_completion_tokens >= _token_limit_warn():
                messages = truncate_history(messages)

            messages.append({"role": "user", "content": prompt_text})

            turn_start = time.time()
            last_tick = time.time()

            # === Streaming через StreamSession ===
            metrics = session.run(
                messages=messages,
                model=_u.MODEL,
            )

            completion_tokens = metrics.completion_tokens
            history_tokens = metrics.prompt_tokens
            chunk_count = metrics.chunk_count
            first_token_time = (
                turn_start + metrics.ttft if metrics.ttft is not None else None
            )
            assistant_content = metrics.assistant_content
            elapsed = metrics.elapsed

            # Live update каждые 1с -- уже завершён, показываем финал
            now = time.time()
            if now - last_tick >= 0.5:
                if completion_tokens > 0:
                    speed = completion_tokens / elapsed if elapsed > 0 else 0
                    final_tokens = completion_tokens
                else:
                    speed = chunk_count / elapsed if elapsed > 0 else 0
                    final_tokens = chunk_count

                ttft = metrics.ttft

                live_parts = f"  {final_tokens:,} tok | {speed:.0f} avg t/s"
                if metrics.instant_speed > 0:
                    live_parts += f" | {metrics.instant_speed:.0f} inst t/s"
                if ttft is not None:
                    live_parts += f" | TTFT {ttft:.2f}s"

                print(f"\r   {live_parts}          ", end="", flush=True)
                last_tick = now

            # Добавляем ответ ассистента в историю
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            # Итоги хода
            wall_total = time.time() - start_wall

            # Защита: если модель ничего не вернула -- пропускаем turn
            if completion_tokens == 0:
                print(f"\r\033[2KCall {turn:>3} | Внимание: пустой ответ, пропускаю...")
                continue

            # Суммируем TTFT
            if metrics.ttft is not None:
                total_ttft += metrics.ttft
                ttft_count += 1

            # chunk.usage.completion_tokens от vLLM = токены текущего запроса
            # Суммируем для кумулятивного Gen
            total_completion_tokens += completion_tokens

            # CallGen = сколько сгенерировали в этом вызове = текущее completion_tokens
            call_gen = completion_tokens

            avg_speed = total_completion_tokens / wall_total if wall_total > 0 else 0
            total_tokens = history_tokens + completion_tokens
            speed = metrics.call_speed

            # === Статичная строка ===
            turn_ttft_str = f" | TTFT {metrics.ttft:.2f}s" if metrics.ttft is not None else ""
            print(f"\r\033[2KCall {turn:>3} | "
                  f"Prompt {history_tokens:,} | "
                  f"Gen {total_completion_tokens:,} | "
                  f"Call {call_gen:,} | "
                  f"Total {total_tokens:,} / {_u.MAX_CONTEXT_TOKENS:,} | "
                  f"{progress_bar(total_tokens, _u.MAX_CONTEXT_TOKENS)} | "
                  f"  {format_time(wall_total)} | "
                  f"Speed {speed:.1f} t/s | "
                  f"Avg {avg_speed:.1f} t/s{turn_ttft_str}")

    except KeyboardInterrupt:
        print("\nПрервано.")
    except Exception as e:
        print(f"\n\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Итоги
    # ------------------------------------------------------------------
    wall_time = time.time() - start_wall
    total_tokens = history_tokens + total_completion_tokens
    avg_speed = total_completion_tokens / wall_time if wall_time > 0 else 0
    avg_ttft = total_ttft / ttft_count if ttft_count > 0 else 0

    print("\n" + "=" * 70)
    print("  ИТОГИ")
    print("=" * 70)
    print(f"  Поворотов:        {turn}")
    print(f"  Prompt (история): {history_tokens:,}")
    print(f"  Completion cum:   {total_completion_tokens:,}")
    print(f"  Всего:            {total_tokens:,}")
    print(f"  Стен-тайм:        {format_time(wall_time)} ({wall_time:.1f}с)")
    print(f"  Avg speed:        {avg_speed:.2f} tok/s")
    print(f"  Avg TTFT:         {avg_ttft:.2f}s")
    print("=" * 70)


def cli():
    """CLI entry point для bench_single."""
    parser = argparse.ArgumentParser(
        prog="bench_single",
        description="Бенчмарк скорости стриминга LLM (последовательный режим)",
    )
    add_common_args(parser)
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Ограничение по времени в секундах (без параметра -- до лимита контекста)",
    )
    args = parser.parse_args()
    run_benchmark(
        duration=args.duration,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_context=args.max_context,
    )


if __name__ == "__main__":
    cli()
