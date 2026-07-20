#!/usr/bin/env python3
"""
tests/test_cli_common.py

Тесты для cli_common.py -- add_common_args(), apply_config().
"""

import argparse

import pytest

from llm_speed_benchmark.cli_common import add_common_args, apply_config


# ============================================================================
# add_common_args()
# ============================================================================

class TestAddCommonArgs:
    """add_common_args: добавление аргументов в парсер."""

    def test_adds_all_args(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([
            "-u", "http://test:8000/v1",
            "-k", "my-key",
            "-m", "my-model",
            "--max-context", "32768",
        ])
        assert args.base_url == "http://test:8000/v1"
        assert args.api_key == "my-key"
        assert args.model == "my-model"
        assert args.max_context == 32768

    def test_defaults_none(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        assert args.base_url is None
        assert args.api_key is None
        assert args.model is None
        assert args.max_context is None

    def test_short_flags(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(["-u", "http://x", "-k", "k", "-m", "m"])
        assert args.base_url == "http://x"
        assert args.api_key == "k"
        assert args.model == "m"

    def test_long_flags(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(["--base-url", "http://y", "--max-context", "65536"])
        assert args.base_url == "http://y"
        assert args.max_context == 65536


# ============================================================================
# apply_config()
# ============================================================================

class TestApplyConfig:
    """apply_config: применение конфигурации к модулю utils."""

    def test_applies_base_url(self):
        import llm_speed_benchmark.utils as _u
        original = _u.BASE_URL
        try:
            apply_config(base_url="http://new:8000/v1")
            assert _u.BASE_URL == "http://new:8000/v1"
        finally:
            _u.BASE_URL = original

    def test_applies_api_key(self):
        import llm_speed_benchmark.utils as _u
        original = _u.API_KEY
        try:
            apply_config(api_key="new-key")
            assert _u.API_KEY == "new-key"
        finally:
            _u.API_KEY = original

    def test_applies_model(self):
        import llm_speed_benchmark.utils as _u
        original = _u.MODEL
        try:
            apply_config(model="new-model")
            assert _u.MODEL == "new-model"
        finally:
            _u.MODEL = original

    def test_applies_max_context(self):
        import llm_speed_benchmark.utils as _u
        original_ctx = _u.MAX_CONTEXT_TOKENS
        original_warn = _u.TOKEN_LIMIT_WARN
        try:
            apply_config(max_context=32768)
            assert _u.MAX_CONTEXT_TOKENS == 32768
            assert _u.TOKEN_LIMIT_WARN == int(32768 * 0.85)
        finally:
            _u.MAX_CONTEXT_TOKENS = original_ctx
            _u.TOKEN_LIMIT_WARN = original_warn

    def test_none_values_no_change(self):
        """None значения не меняют конфигурацию."""
        import llm_speed_benchmark.utils as _u
        original = _u.BASE_URL
        apply_config(base_url=None, api_key=None, model=None, max_context=None)
        assert _u.BASE_URL == original

    def test_partial_apply(self):
        """Частичное применение -- только указанные поля."""
        import llm_speed_benchmark.utils as _u
        original_url = _u.BASE_URL
        original_key = _u.API_KEY
        try:
            apply_config(base_url="http://new:8000/v1")
            assert _u.BASE_URL == "http://new:8000/v1"
            assert _u.API_KEY == original_key  # не изменился
        finally:
            _u.BASE_URL = original_url

    def test_all_at_once(self):
        """Все параметры одновременно."""
        import llm_speed_benchmark.utils as _u
        orig = {
            "url": _u.BASE_URL,
            "key": _u.API_KEY,
            "model": _u.MODEL,
            "ctx": _u.MAX_CONTEXT_TOKENS,
            "warn": _u.TOKEN_LIMIT_WARN,
        }
        try:
            apply_config(
                base_url="http://all:8000/v1",
                api_key="all-key",
                model="all-model",
                max_context=131072,
            )
            assert _u.BASE_URL == "http://all:8000/v1"
            assert _u.API_KEY == "all-key"
            assert _u.MODEL == "all-model"
            assert _u.MAX_CONTEXT_TOKENS == 131072
            assert _u.TOKEN_LIMIT_WARN == int(131072 * 0.85)
        finally:
            _u.BASE_URL = orig["url"]
            _u.API_KEY = orig["key"]
            _u.MODEL = orig["model"]
            _u.MAX_CONTEXT_TOKENS = orig["ctx"]
            _u.TOKEN_LIMIT_WARN = orig["warn"]
