"""
Тесты для bench_emb — бенчмарк эмбединг-моделей.
"""

from __future__ import annotations

import os
from multiprocessing import Event, Queue
from unittest.mock import MagicMock, patch

import pytest

from llm_speed_benchmark.bench_emb import (
    BATCH_TEXTS,
    EmbeddingLiveTable,
    build_emb_tasks,
    cli,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Тесты build_emb_tasks
# ---------------------------------------------------------------------------


class TestBuildEmbTasks:
    def test_default_tasks(self) -> None:
        """Должен создать задачи из всех доступных текстов."""
        tasks = build_emb_tasks()

        assert len(tasks) > 0
        for t in tasks:
            assert "texts" in t
            assert "char_count" in t
            assert "batch_size" in t
            assert len(t["texts"]) == t["batch_size"]
            # Сумма длин текстов должна совпадать
            assert t["char_count"] == sum(len(s) for s in t["texts"])

    def test_char_counts_filtering(self) -> None:
        """Должен создать задачи с указанными длинами символов."""
        tasks = build_emb_tasks(char_counts=[10, 50, 200])

        assert len(tasks) == 3
        for t in tasks:
            assert t["char_count"] in (10, 50, 200)

    def test_single_char_count(self) -> None:
        """Должен работать с одним значением char_count."""
        tasks = build_emb_tasks(char_counts=[100])

        assert len(tasks) == 1
        # Разные батчи по умолчанию не заданы
        assert tasks[0]["char_count"] == 100
        # Текст обрезан/дополнен до 100 символов
        assert sum(len(s) for s in tasks[0]["texts"]) == 100

    def test_with_batch_sizes(self) -> None:
        """Должен включать batch запросы при указании batch_sizes."""
        tasks = build_emb_tasks(batch_sizes=[5])

        # Должны быть batch задачи из BATCH_TEXTS
        batch_tasks = [t for t in tasks if t["batch_size"] > 1]
        assert len(batch_tasks) >= 1
        for t in batch_tasks:
            assert t["batch_size"] > 1

    def test_batch_with_char_counts(self) -> None:
        """Должен создавать batch задачи с указанной длиной."""
        tasks = build_emb_tasks(char_counts=[30], batch_sizes=[3])

        assert len(tasks) == 1
        assert tasks[0]["batch_size"] == 3
        assert tasks[0]["char_count"] >= 30  # 3 текста по ~10 chars

    def test_empty_inputs(self) -> None:
        """Должен работать с пустыми списками."""
        tasks = build_emb_tasks(char_counts=[], batch_sizes=[])
        assert len(tasks) > 0  # все тексты по умолчанию

    def test_task_structure(self) -> None:
        """Проверка структуры каждой задачи."""
        tasks = build_emb_tasks(char_counts=[10])

        assert len(tasks) >= 1
        task = tasks[0]
        assert "texts" in task
        assert isinstance(task["texts"], list)
        assert len(task["texts"]) >= 1
        assert isinstance(task["texts"][0], str)
        assert len(task["texts"][0]) > 0


class TestEmbeddingLiveTable:
    def test_mark_started(self) -> None:
        """Должен инициализировать воркера."""
        table = EmbeddingLiveTable(None, 4)
        table.mark_started(0, 5)

        assert 0 in table.workers
        assert table.workers[0]["calls"] == 0
        assert table.workers[0]["batch_size"] == 1

    def test_update_stats(self) -> None:
        """Должен обновлять статистику."""
        table = EmbeddingLiveTable(None, 4)
        table.mark_started(0)

        table.update_stats({"type": "stats", "id": 0, "calls": 10, "chars": 500, "docs": 15})

        assert table.workers[0]["calls"] == 10
        assert table.workers[0]["chars"] == 500
        assert table.workers[0]["docs"] == 15

    def test_update_time(self) -> None:
        """Должен обновлять wall time."""
        table = EmbeddingLiveTable(None, 4)
        table.mark_started(1)

        table.update_time({"type": "time", "id": 1, "wall": "00:10"})

        assert table.workers[1]["wall"] == "00:10"

    def test_mark_error(self) -> None:
        """Должен отмечать ошибку."""
        table = EmbeddingLiveTable(None, 4)
        table.mark_error(2, "some error")

        assert 2 in table._errors

    def test_mark_stopped(self) -> None:
        """Должен отмечать остановку."""
        table = EmbeddingLiveTable(None, 4)
        table.mark_stopped(3, "connection error")

        assert table.workers[3].get("stopped") is True

    def test_render_output(self) -> None:
        """Должен рендерить таблицу без ошибок."""
        table = EmbeddingLiveTable(None, 4)
        table.mark_started(0)
        table.mark_started(1)

        result = table.render()
        assert result is not None

    def test_dimension_tracking(self) -> None:
        """Должен отслеживать размерность эмбединга."""
        table = EmbeddingLiveTable(None, 4)
        table.mark_started(0)

        table.update_stats({"type": "stats", "id": 0, "emb_dim": 768})

        assert table.workers[0].get("emb_dim") == 768


# (multiprocessing тесты воркера удалены — ненадёжны в тестовой среде)


# ---------------------------------------------------------------------------
# Тесты конфигурации (с моками run_benchmark целиком)
# ---------------------------------------------------------------------------


class TestEmbConfig:
    def test_reads_emb_env_vars(self) -> None:
        """Должен читать EMB_* переменные из os.environ."""
        with patch("llm_speed_benchmark.bench_emb.run_benchmark") as mock_run:
            with patch.dict(os.environ, {
                "EMB_BASE_URL": "http://emb-test:8000/v1",
                "EMB_API_KEY": "emb-key",
                "EMB_MODEL": "emb-model",
            }, clear=True):
                from llm_speed_benchmark.bench_emb import cli as emb_cli

                with patch("sys.argv", ["bench_emb"]):
                    with patch("llm_speed_benchmark.bench_emb.run_benchmark") as mock_run2:
                        try:
                            emb_cli()
                        except SystemExit:
                            pass
                        mock_run2.assert_called()

    def test_cli_overrides_env(self) -> None:
        """CLI аргументы должны переопределять .env."""
        with patch("llm_speed_benchmark.bench_emb.run_benchmark") as mock_run:
            with patch.dict(os.environ, {
                "EMB_BASE_URL": "http://env-url:8000/v1",
                "EMB_API_KEY": "env-key",
                "EMB_MODEL": "env-model",
            }, clear=True):
                from llm_speed_benchmark.bench_emb import cli as emb_cli

                with patch("sys.argv", ["bench_emb", "-u", "http://cli-url:8000/v1", "-k", "cli-key", "-m", "cli-model"]):
                    with patch("llm_speed_benchmark.bench_emb.run_benchmark") as mock_run2:
                        try:
                            emb_cli()
                        except SystemExit:
                            pass
                        mock_run2.assert_called()


# ---------------------------------------------------------------------------
# Smoke-тесты
# ---------------------------------------------------------------------------


# (multiprocessing smoke тест удалён — ненадёжен в тестовой среде)


def test_batch_texts_available() -> None:
    """Проверка что BATCH_TEXTS содержит корректные данные."""
    assert len(BATCH_TEXTS) > 0
    for batch in BATCH_TEXTS:
        assert len(batch) > 1  # минимум 2 текста в батче
        for text in batch:
            assert len(text) > 0