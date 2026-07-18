#!/usr/bin/env python3
"""
tests/test_long_context.py

Тесты для long_context.py -- LongContextDataset.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_speed_benchmark.long_context import LongContextDataset


# ============================================================================
# LongContextDataset -- инициализация
# ============================================================================

class TestLongContextInit:
    """LongContextDataset: инициализация и свойства."""

    def test_default_init(self):
        ds = LongContextDataset()
        assert ds.dataset_name == "beam"
        assert not ds.is_downloaded

    def test_custom_dataset(self):
        ds = LongContextDataset(dataset_name="longbench-chat")
        assert ds.dataset_name == "longbench-chat"

    def test_cache_dir_expanded(self, tmp_path):
        ds = LongContextDataset(cache_dir=str(tmp_path))
        assert ds.cache_dir == tmp_path

    def test_data_dir(self, tmp_path):
        ds = LongContextDataset(cache_dir=str(tmp_path))
        assert ds.data_dir == tmp_path / "beam"

    def test_is_downloaded_false(self):
        ds = LongContextDataset()
        assert not ds.is_downloaded

    def test_is_downloaded_true(self, tmp_path):
        data_dir = tmp_path / "beam"
        data_dir.mkdir()
        (data_dir / "test.json").write_text("[]")
        ds = LongContextDataset(cache_dir=str(tmp_path))
        assert ds.is_downloaded


# ============================================================================
# LongContextDataset -- _estimate_text_tokens
# ============================================================================

class TestEstimateTokens:
    """LongContextDataset: оценка токенов."""

    def test_empty(self):
        assert LongContextDataset._estimate_text_tokens("") == 0

    def test_short_text(self):
        tokens = LongContextDataset._estimate_text_tokens("hello")
        assert tokens >= 1

    def test_long_text(self):
        text = "a" * 120  # ~40 токенов
        tokens = LongContextDataset._estimate_text_tokens(text)
        assert 35 <= tokens <= 45

    def test_cyrillic(self):
        tokens = LongContextDataset._estimate_text_tokens("Привет мир")
        assert tokens >= 1


# ============================================================================
# LongContextDataset -- load с фикстурами
# ============================================================================

class TestLongContextLoad:
    """LongContextDataset: загрузка из локальных файлов."""

    def _create_beam_fixture(self, tmp_path):
        """Создаёт фикстуру BEAM данных."""
        data_dir = tmp_path / "beam"
        data_dir.mkdir()
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there"},
                    {"from": "human", "value": "How are you"},
                    {"from": "gpt", "value": "I am fine"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Question one"},
                    {"from": "gpt", "value": "Answer one"},
                ]
            },
        ]
        (data_dir / "conv_0.json").write_text(json.dumps(records[0]))
        (data_dir / "conv_1.json").write_text(json.dumps(records[1]))
        return data_dir

    def test_load_beam(self, tmp_path):
        self._create_beam_fixture(tmp_path)
        ds = LongContextDataset(cache_dir=str(tmp_path))
        data = ds.load()
        assert len(data) == 2

    def test_load_not_downloaded(self):
        ds = LongContextDataset()
        with pytest.raises(FileNotFoundError):
            ds.load()

    def test_load_cached(self, tmp_path):
        self._create_beam_fixture(tmp_path)
        ds = LongContextDataset(cache_dir=str(tmp_path))
        data1 = ds.load()
        data2 = ds.load()  # Должен вернуть кэш
        assert data1 is data2

    def test_unknown_dataset(self, tmp_path):
        # Создаём пустую директорию чтобы is_downloaded=True
        data_dir = tmp_path / "unknown"
        data_dir.mkdir()
        (data_dir / "dummy.json").write_text("[]")
        ds = LongContextDataset(dataset_name="unknown", cache_dir=str(tmp_path))
        with pytest.raises(ValueError):
            ds.load()


# ============================================================================
# LongContextDataset -- get_messages
# ============================================================================

class TestLongContextMessages:
    """LongContextDataset: преобразование в OpenAI messages."""

    def _create_beam_fixture(self, tmp_path):
        data_dir = tmp_path / "beam"
        data_dir.mkdir()
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello world"},
                    {"from": "gpt", "value": "Hi there friend"},
                    {"from": "human", "value": "Tell me more"},
                    {"from": "gpt", "value": "Sure thing"},
                ]
            },
        ]
        (data_dir / "conv_0.json").write_text(json.dumps(records[0]))
        return data_dir

    def test_beam_to_messages(self, tmp_path):
        self._create_beam_fixture(tmp_path)
        ds = LongContextDataset(cache_dir=str(tmp_path))
        messages = ds.get_messages(conversation_id=0)
        assert messages[0]["role"] == "system"
        # После system: human -> user, gpt -> assistant
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello world"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there friend"

    def test_beam_to_messages_roles(self, tmp_path):
        self._create_beam_fixture(tmp_path)
        ds = LongContextDataset(cache_dir=str(tmp_path))
        messages = ds.get_messages(conversation_id=0)
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_out_of_range(self, tmp_path):
        self._create_beam_fixture(tmp_path)
        ds = LongContextDataset(cache_dir=str(tmp_path))
        with pytest.raises(IndexError):
            ds.get_messages(conversation_id=99)

    def test_max_tokens_truncation(self, tmp_path):
        data_dir = tmp_path / "beam"
        data_dir.mkdir()
        # Создаём длинную conversation
        convs = []
        for i in range(100):
            convs.append({"from": "human", "value": f"Message number {i} with some text"})
            convs.append({"from": "gpt", "value": f"Response number {i} with more text"})
        records = [{"conversations": convs}]
        (data_dir / "long.json").write_text(json.dumps(records[0]))

        ds = LongContextDataset(cache_dir=str(tmp_path))
        messages = ds.get_messages(conversation_id=0, max_tokens=50)
        # Должно быть обрезано -- меньше 100 сообщений
        assert len(messages) < 100
        # Но минимум system + 1
        assert len(messages) >= 2


# ============================================================================
# LongContextDataset -- get_conversations
# ============================================================================

class TestLongContextFilter:
    """LongContextDataset: фильтрация по длине."""

    def test_filter_by_tokens(self, tmp_path):
        data_dir = tmp_path / "beam"
        data_dir.mkdir()
        # Короткая conversation (~5 токенов)
        short = {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Ok"}]}
        # Длинная conversation (~50 токенов)
        long_text = " ".join([f"word{i}" for i in range(150)])
        long = {"conversations": [{"from": "human", "value": long_text}]}
        (data_dir / "short.json").write_text(json.dumps(short))
        (data_dir / "long.json").write_text(json.dumps(long))

        ds = LongContextDataset(cache_dir=str(tmp_path))
        # Фильтр 50-500 токенов -- должна попасть длинная conversation
        filtered = ds.get_conversations(min_tokens=50, max_tokens=500)
        assert len(filtered) >= 1
        for f in filtered:
            assert 50 <= f["estimated_tokens"] <= 500


# ============================================================================
# LongContextDataset -- info()
# ============================================================================

class TestLongContextInfo:
    """LongContextDataset: info()."""

    def test_info_not_downloaded(self):
        ds = LongContextDataset()
        info = ds.info()
        assert info["dataset"] == "beam"
        assert info["downloaded"] is False

    def test_info_downloaded(self, tmp_path):
        data_dir = tmp_path / "beam"
        data_dir.mkdir()
        record = {"conversations": [{"from": "human", "value": "Hello world test"}]}
        (data_dir / "conv.json").write_text(json.dumps(record))

        ds = LongContextDataset(cache_dir=str(tmp_path))
        info = ds.info()
        assert info["downloaded"] is True
        assert info["records"] == 1
        assert "min_tokens" in info
        assert "max_tokens" in info
        assert "avg_tokens" in info


# ============================================================================
# LongContextDataset -- download (моки)
# ============================================================================

class TestLongContextDownload:
    """LongContextDataset: download() с моками."""

    def test_download_beam_already_exists(self, tmp_path, capsys):
        data_dir = tmp_path / "beam"
        data_dir.mkdir()
        (data_dir / "existing.json").write_text("[]")
        ds = LongContextDataset(cache_dir=str(tmp_path))
        ds.download()
        captured = capsys.readouterr()
        assert "уже скачан" in captured.out

    def test_download_unknown_dataset(self):
        ds = LongContextDataset(dataset_name="unknown")
        with pytest.raises(ValueError):
            ds.download()

    def test_download_beam_success(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="done")
            ds = LongContextDataset(cache_dir=str(tmp_path))
            ds.download()
            mock_run.assert_called_once()
            # Проверяем, что в команде есть repo_id
            call_args = mock_run.call_args[0][0]
            assert "Mohammadta/BEAM" in call_args

    def test_download_beam_failure(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="auth failed", stdout="")
            ds = LongContextDataset(cache_dir=str(tmp_path))
            with pytest.raises(RuntimeError, match="auth failed"):
                ds.download()
