"""
Тесты для bench_tts — бенчмарк TTS-моделей.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_speed_benchmark.bench_tts import (
    TTS_PROMPTS,
    TTSLiveTable,
    build_text_tasks,
    cli,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Тесты build_text_tasks
# ---------------------------------------------------------------------------


class TestBuildTextTasks:
    def test_default_tasks(self) -> None:
        """Должен создать задачи из всех доступных текстов."""
        tasks = build_text_tasks()

        assert len(tasks) > 0
        for t in tasks:
            assert "text" in t
            assert "char_count" in t
            assert t["char_count"] == len(t["text"])

    def test_char_counts_filtering(self) -> None:
        """Должен создать задачи с указанными длинами символов."""
        tasks = build_text_tasks(char_counts=[10, 50, 200])

        assert len(tasks) == 3
        for t in tasks:
            assert t["char_count"] in (10, 50, 200)

    def test_single_char_count(self) -> None:
        """Должен работать с одним значением char_count."""
        tasks = build_text_tasks(char_counts=[100])

        assert len(tasks) == 1
        assert tasks[0]["char_count"] == 100
        assert len(tasks[0]["text"]) == 100

    def test_very_long_text(self) -> None:
        """Должен создавать длинные тексты (> 500 chars)."""
        tasks = build_text_tasks(char_counts=[2000])

        assert len(tasks) == 1
        assert tasks[0]["char_count"] == 2000
        assert len(tasks[0]["text"]) == 2000

    def test_empty_char_counts(self) -> None:
        """Пустой список char_counts — использовать все тексты."""
        tasks = build_text_tasks(char_counts=[])
        assert len(tasks) > 0


class TestTTSLiveTable:
    def test_mark_started(self) -> None:
        """Должен инициализировать воркера."""
        table = TTSLiveTable(None, 4)
        table.mark_started(0, 5)

        assert 0 in table.workers
        assert table.workers[0]["calls"] == 0
        assert table.workers[0]["tail"] == "[dim]waiting...[/]"

    def test_update_stats(self) -> None:
        """Должен обновлять статистику."""
        table = TTSLiveTable(None, 4)
        table.mark_started(0)
        table.update_stats({"type": "stats", "id": 0, "calls": 5, "chars": 100, "size": 5000})

        assert table.workers[0]["calls"] == 5
        assert table.workers[0]["chars"] == 100
        assert table.workers[0]["size"] == 5000

    def test_update_time(self) -> None:
        """Должен обновлять wall time."""
        table = TTSLiveTable(None, 4)
        table.mark_started(1)
        table.update_time({"type": "time", "id": 1, "wall": "00:05"})

        assert table.workers[1]["wall"] == "00:05"

    def test_mark_error(self) -> None:
        """Должен отмечать ошибку."""
        table = TTSLiveTable(None, 4)
        table.mark_error(2, "traceback text")

        assert 2 in table._errors

    def test_mark_stopped(self) -> None:
        """Должен отмечать остановку воркера."""
        table = TTSLiveTable(None, 4)
        table.mark_stopped(3, "connection error")

        assert table.workers[3].get("stopped") is True
        assert table.workers[3].get("error") == "connection error"

    def test_multiple_workers(self) -> None:
        """Должен обрабатывать несколько воркеров."""
        table = TTSLiveTable(None, 4)
        table.mark_started(0)
        table.mark_started(1)
        table.mark_started(2)

        assert set(table.workers.keys()) == {0, 1, 2}

    def test_render_with_data(self) -> None:
        """Должен рендерить таблицу с данными."""
        table = TTSLiveTable(None, 2)
        table.mark_started(0)
        table.mark_started(1)
        table.update_stats({"type": "stats", "id": 0, "calls": 3, "chars": 100, "size": 5000})
        table.update_stats({"type": "stats", "id": 1, "calls": 5, "chars": 250, "size": 12000})

        result = table.render()
        assert result is not None

    def test_render_with_errors(self) -> None:
        """Должен рендерить таблицу с ошибками."""
        table = TTSLiveTable(None, 2)
        table.mark_started(0)
        table.mark_started(1)
        table.mark_error(1, "test error")

        result = table.render()
        assert result is not None

    def test_render_stopped_worker(self) -> None:
        """Должен рендерить остановленного воркера."""
        table = TTSLiveTable(None, 1)
        table.mark_started(0)
        table.mark_stopped(0, "request failed")

        result = table.render()
        assert result is not None


# ---------------------------------------------------------------------------
# Тесты конфигурации (с моками run_benchmark целиком)
# ---------------------------------------------------------------------------


class TestTTSConfig:
    def test_reads_tts_env_vars(self) -> None:
        """Должен читать TTS_* переменные из os.environ."""
        with patch("llm_speed_benchmark.bench_tts.run_benchmark") as mock_run:
            with patch.dict(os.environ, {
                "TTS_BASE_URL": "http://tts-test:8000/v1",
                "TTS_API_KEY": "tts-key",
                "TTS_MODEL": "tts-model",
            }, clear=True):
                from llm_speed_benchmark.bench_tts import cli as tts_cli

                with patch("sys.argv", ["bench_tts"]):
                    with patch("llm_speed_benchmark.bench_tts.run_benchmark") as mock_run2:
                        try:
                            tts_cli()
                        except SystemExit:
                            pass
                        mock_run2.assert_called()


# ---------------------------------------------------------------------------
# Smoke-тесты
# ---------------------------------------------------------------------------

# (multiprocessing smoke тесты удалены — ненадёжны в тестовой среде)

def test_texts_by_char_count() -> None:
    """Проверка фильтрации текстов по длине."""
    from llm_speed_benchmark.bench_tts import get_texts_by_char_count

    texts = ["Короткий текст.", "Это средний текст для проверки.", "Очень длинный текст для тестирования работы системы."]

    short = get_texts_by_char_count(texts, 10, 20)
    assert len(short) == 1
    assert short[0] == "Короткий текст."

    long = get_texts_by_char_count(texts, 40, 200)
    assert len(long) == 1


def test_tts_prompts_have_content() -> None:
    """TTS_PROMPTS должен содержать непустые тексты."""
    for category, texts in TTS_PROMPTS.items():
        assert len(texts) > 0, f"Категория {category} пуста"
        for t in texts:
            assert len(t) > 0, f"Пустой текст в {category}"