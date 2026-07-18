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
    from llm_speed_benchmark.utils import _token_limit_warn

    if base_url is not None:
        _u.BASE_URL = base_url
    if api_key is not None:
        _u.API_KEY = api_key
    if model is not None:
        _u.MODEL = model
    if max_context is not None:
        _u.MAX_CONTEXT_TOKENS = max_context
        _u.TOKEN_LIMIT_WARN = _token_limit_warn()
