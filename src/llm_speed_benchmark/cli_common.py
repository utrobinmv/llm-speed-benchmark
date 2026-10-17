"""
llm_speed_benchmark/cli_common.py

Общие CLI аргументы и применение конфигурации.
"""

from __future__ import annotations

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Добавляет общие аргументы CLI (--base-url, --api-key, --model, --max-context)."""
    parser.add_argument(
        "--base-url", "-u", type=str, default=None,
        help="Адрес API (OpenAI-compatible)",
    )
    parser.add_argument(
        "--api-key", "-k", type=str, default=None,
        help="API ключ",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Название модели",
    )
    parser.add_argument(
        "--max-context", type=int, default=None,
        help="Переопределение MAX_CONTEXT_TOKENS",
    )


def apply_config(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_context: int | None = None,
) -> None:
    """Применяет CLI аргументы к модулю utils (глобальные переменные).

    Приоритет: CLI > .env
    """
    import llm_speed_benchmark.utils as _u  # noqa: PLC0414

    # Пустая/только-пробельная строка (-–base-url " ") считается "не задано"
    # и НЕ затирает реальное значение из .env: иначе она молча ломала бы
    # подключение, заменяя настроенный base_url/api_key/model на мусор.
    if base_url and base_url.strip():
        _u.BASE_URL = base_url
    if api_key and api_key.strip():
        _u.API_KEY = api_key
    if model and model.strip():
        _u.MODEL = model
    if max_context is not None:
        # Единый сеттер держит MAX_CONTEXT_TOKENS и TOKEN_LIMIT_WARN в синхроне.
        _u.set_max_context(max_context)
