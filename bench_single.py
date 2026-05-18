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

from dotenv import load_dotenv
from openai import OpenAI

API_KEY = os.getenv("API_KEY", "sk-vllm-qwen3.5-0.8b")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "qwen3.5-0.8b")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "262144"))

TOKEN_LIMIT_WARN = int(MAX_CONTEXT_TOKENS * 0.85)

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def get_client():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0)


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

def run_benchmark():
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
    print("=" * 70)
    print()

    total_completion_tokens = 0  # cumulative completion tokens from API
    history_tokens = 0
    prev_completion = 0
    start_wall = time.time()
    turn = 0
    last_stats_time = time.time()

    try:
        while True:
            turn += 1
            prompt_text = prompts[0 if turn == 1 else 1]

            # Проверка контекста
            if history_tokens + total_completion_tokens >= MAX_CONTEXT_TOKENS:
                break

            if history_tokens + total_completion_tokens >= TOKEN_LIMIT_WARN:
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

            for chunk in response:
                if chunk.usage:
                    history_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0

                # Собираем контент ассистента для истории
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta is not None:
                        content = getattr(delta, 'content', None) or ""
                        assistant_content += content

                # Live update каждые 1с
                now = time.time()
                if now - last_tick >= 1.0:
                    elapsed = now - turn_start
                    if completion_tokens > 0:
                        speed = completion_tokens / elapsed if elapsed > 0 else 0
                        est_tokens = completion_tokens
                    else:
                        speed = 300  # оценочная скорость
                        est_tokens = int(elapsed * speed)
                    print(f"\r   ⏱ {est_tokens:,} tok | {speed:.0f} tok/s          ", end="", flush=True)
                    last_tick = now

            # После завершения стриминга — последнее обновление live
            now = time.time()
            if now - last_tick >= 0.5:
                elapsed_live = now - turn_start
                if completion_tokens > 0:
                    speed_live = completion_tokens / elapsed_live
                else:
                    speed_live = 300
                print(f"\r   ⏱ {completion_tokens:,} tok | {speed_live:.0f} tok/s          ", end="", flush=True)
                last_tick = now

            # Добавляем ответ ассистента в историю
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            # Итоги хода
            elapsed = time.time() - turn_start
            wall_total = time.time() - start_wall

            # Если usage не пришло — фолбэк на оценку
            if completion_tokens == 0:
                completion_tokens = int(elapsed * 300)

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
            print(f"\r\033[2KCall {turn:>3} | "
                  f"Prompt {history_tokens:,} | "
                  f"Gen {total_completion_tokens:,} | "
                  f"Call {call_gen:,} | "
                  f"Total {total_tokens:,} / {MAX_CONTEXT_TOKENS:,} | "
                  f"{progress_bar(total_tokens, MAX_CONTEXT_TOKENS)} | "
                  f"⏱ {format_time(wall_total)} | "
                  f"Last {speed:.1f} | "
                  f"Avg {avg_speed:.1f} tok/s")

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

    print("\n" + "=" * 70)
    print("  ИТОГИ")
    print("=" * 70)
    print(f"  Поворотов:        {turn}")
    print(f"  Prompt (история): {history_tokens:,}")
    print(f"  Completion cum:   {total_completion_tokens:,}")
    print(f"  Всего:            {total_tokens:,}")
    print(f"  Стен-тайм:        {format_time(wall_time)} ({wall_time:.1f}с)")
    print(f"  Avg speed:        {avg_speed:.2f} tok/s")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
