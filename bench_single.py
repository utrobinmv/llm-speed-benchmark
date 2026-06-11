#!/usr/bin/env python3
"""
bench_single.py

Бенчмарк скорости vLLM в режиме стриминга. Не выводит текст ответа,
только статистику. Работает до заполнения контекстного окна.

Использование:
  source .venv
  python3 bench_single.py
"""

import os
import sys
import time
import argparse

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("API_KEY", "sk-vllm-qwen3.5-0.8b")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "qwen3.5-0.8b")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "262144"))

def _token_limit_warn():
    """85% от MAX_CONTEXT_TOKENS — пересчитывается динамически."""
    return int(MAX_CONTEXT_TOKENS * 0.85)


TOKEN_LIMIT_WARN = _token_limit_warn()

# Окно для мгновенной скорости (последние N токенов)
INSTANT_WINDOW = 200

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def get_client():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)


def truncate_history(messages):
    """Удаляем самые старые пары user/assistant, оставляем system."""
    result = list(messages)
    # Удаляем пары пока не останется только system + 1 пара
    while len(result) > 3:
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


def progress_bar(context_tokens, max_tokens, width=25):
    if max_tokens == 0:
        return "[" + "░" * width + "] 0.0%"
    filled = min(int(width * context_tokens / max_tokens), width)
    bar = "█" * filled + "░" * (width - filled)
    pct = min(100.0 * context_tokens / max_tokens, 100.0)
    return f"[{bar}] {pct:.1f}%"


def format_time(seconds):
    """Форматирует секунды в М:СС."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run_benchmark(duration=None):
    client = get_client()

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
    print(f"  Модель:         {MODEL}")
    print(f"  Max context:    {MAX_CONTEXT_TOKENS:,}")
    print(f"  Предупреждение: {TOKEN_LIMIT_WARN:,} (85%)")
    print(f"  Instant window: {INSTANT_WINDOW} tok")
    if duration is not None:
        print(f"  Длительность:   {duration}с")
    print("=" * 70)
    print()

    total_completion_tokens = 0  # cumulative completion tokens from API
    history_tokens = 0
    prev_completion = 0
    start_wall = time.time()
    turn = 0
    last_stats_time = time.time()

    # Суммарный TTFT (Time To First Token) по всем turn'ам
    total_ttft = 0.0
    ttft_count = 0

    try:
        while True:
            # Проверка по времени (duration is not None — корректно для 0)
            if duration is not None and (time.time() - start_wall) >= duration:
                break

            turn += 1
            prompt_text = prompts[0 if turn == 1 else 1]

            # Проверка контекста
            if history_tokens + total_completion_tokens >= MAX_CONTEXT_TOKENS:
                break

            if history_tokens + total_completion_tokens >= _token_limit_warn():
                messages = truncate_history(messages)

            messages.append({"role": "user", "content": prompt_text})

            turn_start = time.time()
            last_tick = time.time()

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=8192,
                stream_options={"include_usage": True},
            )

            # === Извлекаем completion_tokens и собираем ответ ассистента ===
            completion_tokens = 0
            history_tokens = 0
            assistant_content = ""
            chunk_count = 0  # ≈ токены в реальном времени (1 чанк ≈ 1 токен)
            first_token_time = None  # время прихода первого токена

            # Для мгновенной скорости: (timestamp, cumulative_chunk) для каждого чанка
            instant_buffer = []

            for chunk in response:
                if chunk.usage:
                    history_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0

                # Собираем контент ассистента для истории
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

                # Live update каждые 1с — считаем реальные чанки
                now = time.time()
                if now - last_tick >= 1.0:
                    elapsed = now - turn_start
                    if completion_tokens > 0:
                        # Usage пришло (финальный чанк) — точная цифра
                        est_tokens = completion_tokens
                    elif chunk_count > 0:
                        # Стрим идёт — считаем чанки как ≈ токены
                        est_tokens = chunk_count
                    else:
                        # Ещё ничего не пришло — не выводим мусор
                        est_tokens = 0

                    # Мгновенная скорость: последние INSTANT_WINDOW токенов
                    instant_speed = 0
                    if len(instant_buffer) > INSTANT_WINDOW:
                        window_start = instant_buffer[-INSTANT_WINDOW][0]
                        window_end = instant_buffer[-1][0]
                        window_tokens = instant_buffer[-1][1] - instant_buffer[-INSTANT_WINDOW - 1][1] if len(instant_buffer) > INSTANT_WINDOW + 1 else INSTANT_WINDOW
                        window_time = window_end - window_start
                        instant_speed = window_tokens / window_time if window_time > 0 else 0

                    # TTFT для этого turn'а
                    ttft = (first_token_time - turn_start) if first_token_time is not None else None

                    # Форматируем live строку
                    if completion_tokens > 0:
                        speed = completion_tokens / elapsed if elapsed > 0 else 0
                    elif chunk_count > 0:
                        speed = chunk_count / elapsed if elapsed > 0 else 0
                    else:
                        speed = 0

                    live_parts = f"⏱ {est_tokens:,} tok | {speed:.0f} avg t/s"
                    if instant_speed > 0:
                        live_parts += f" | {instant_speed:.0f} inst t/s"
                    if ttft is not None:
                        live_parts += f" | TTFT {ttft:.2f}s"

                    print(f"\r   {live_parts}          ", end="", flush=True)
                    last_tick = now

            # После завершения стриминга — последнее обновление live
            now = time.time()
            if now - last_tick >= 0.5:
                elapsed_live = now - turn_start
                if completion_tokens > 0:
                    # Точная цифра из usage
                    speed_live = completion_tokens / elapsed_live
                    final_tokens = completion_tokens
                else:
                    # Fallback на чанки (если usage не пришло)
                    speed_live = chunk_count / elapsed_live if elapsed_live > 0 else 0
                    final_tokens = chunk_count

                # Мгновенная скорость
                instant_speed = 0
                if len(instant_buffer) > INSTANT_WINDOW:
                    window_start = instant_buffer[-INSTANT_WINDOW][0]
                    window_end = instant_buffer[-1][0]
                    window_tokens = instant_buffer[-1][1] - instant_buffer[-INSTANT_WINDOW - 1][1] if len(instant_buffer) > INSTANT_WINDOW + 1 else INSTANT_WINDOW
                    window_time = window_end - window_start
                    instant_speed = window_tokens / window_time if window_time > 0 else 0

                ttft = (first_token_time - turn_start) if first_token_time is not None else None

                live_parts = f"⏱ {final_tokens:,} tok | {speed_live:.0f} avg t/s"
                if instant_speed > 0:
                    live_parts += f" | {instant_speed:.0f} inst t/s"
                if ttft is not None:
                    live_parts += f" | TTFT {ttft:.2f}s"

                print(f"\r   {live_parts}          ", end="", flush=True)
                last_tick = now

            # Добавляем ответ ассистента в историю
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            # Итоги хода
            elapsed = time.time() - turn_start
            wall_total = time.time() - start_wall

            # Если usage не пришло — используем счётчик чанков (≈ токены)
            if completion_tokens == 0:
                completion_tokens = chunk_count

            # Защита: если модель ничего не вернула — пропускаем turn
            if completion_tokens == 0:
                print(f"\r\033[2KCall {turn:>3} | ⚠️ Пустой ответ, пропускаю...")
                continue

            # Суммируем TTFT
            if first_token_time is not None:
                turn_ttft = first_token_time - turn_start
                total_ttft += turn_ttft
                ttft_count += 1

            # chunk.usage.completion_tokens от vLLM = токены текущего запроса
            # Суммируем для кумулятивного Gen
            total_completion_tokens += completion_tokens

            # CallGen = сколько сгенерировали в этом вызове = текущее completion_tokens
            call_gen = completion_tokens
            prev_completion = total_completion_tokens

            avg_speed = total_completion_tokens / wall_total if wall_total > 0 else 0
            total_tokens = history_tokens + completion_tokens
            speed = call_gen / elapsed if elapsed > 0 else 0

            # === Статичная строка ===
            turn_ttft_str = f" | TTFT {turn_ttft:.2f}s" if first_token_time is not None else ""
            print(f"\r\033[2KCall {turn:>3} | "
                  f"Prompt {history_tokens:,} | "
                  f"Gen {total_completion_tokens:,} | "
                  f"Call {call_gen:,} | "
                  f"Total {total_tokens:,} / {MAX_CONTEXT_TOKENS:,} | "
                  f"{progress_bar(total_tokens, MAX_CONTEXT_TOKENS)} | "
                  f"⏱ {format_time(wall_total)} | "
                  f"Speed {speed:.1f} t/s | "
                  f"Avg {avg_speed:.1f} t/s{turn_ttft_str}")

    except KeyboardInterrupt:
        print("\n⏹ Прервано.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бенчмарк скорости стриминга LLM")
    parser.add_argument("--duration", type=int, default=None,
                        help="Ограничение по времени в секундах (без параметра — до лимита контекста)")
    args = parser.parse_args()
    run_benchmark(duration=args.duration)
