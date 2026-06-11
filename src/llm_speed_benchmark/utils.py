"""
llm_speed_benchmark/utils.py

Общие утилиты для bench_single и bench_multi.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Загружаем .env из текущей директории и из директории скрипта
load_dotenv()
load_dotenv(os.path.expanduser("~/.llm-speed-benchmark.env"))

API_KEY = os.getenv("API_KEY", "sk-vllm-qwen3.5-0.8b")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "qwen3.5-0.8b")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "262144"))

# Окно для мгновенной скорости (последние N токенов)
INSTANT_WINDOW = 200


def _token_limit_warn():
    """85% от MAX_CONTEXT_TOKENS — пересчитывается динамически."""
    return int(MAX_CONTEXT_TOKENS * 0.85)


TOKEN_LIMIT_WARN = _token_limit_warn()


def get_client():
    """Создаёт OpenAI-совместимый клиент."""
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
