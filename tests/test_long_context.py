"""
tests/test_long_context.py

Тесты для модуля long_context.py -- загрузка и подготовка датасетов длинных контекстов.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from llm_speed_benchmark.long_context import (
    ConversationInfo,
    LongContextDataset,
    _DEFAULT_DATA_DIR,
)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> str:
    """Создаёт временную директорию для тестовых данных."""
    return str(tmp_path / "datasets")


def _make_chat_array(groups: list[list[dict]]) -> np.ndarray:
    """Создаёт numpy array of numpy arrays (как в реальном BEAM)."""
    return np.array(
        [np.array(group, dtype=object) for group in groups], dtype=object
    )


@pytest.fixture
def mock_beam_parquet(tmp_data_dir: str):
    """Мокает pd.read_parquet для возврата тестовых данных BEAM."""
    beam_dir = Path(tmp_data_dir) / "beam" / "data"
    beam_dir.mkdir(parents=True, exist_ok=True)

    # Создаём dummy parquet файл чтобы glob нашёл его
    (beam_dir / "100K-00000-of-00001.parquet").touch()

    chat1 = _make_chat_array([
        [
            {"content": "Hello, can you help me?", "role": "user", "id": 0},
            {"content": "Of course! How can I help?", "role": "assistant", "id": 1},
        ],
        [
            {"content": "Tell me about Python.", "role": "user", "id": 2},
            {"content": "Python is a programming language.", "role": "assistant", "id": 3},
        ],
    ])

    chat2 = _make_chat_array([
        [
            {"content": "A" * 3000, "role": "user", "id": 0},
            {"content": "B" * 3000, "role": "assistant", "id": 1},
        ],
    ])

    mock_df = pd.DataFrame([
        {
            "conversation_id": 1,
            "chat": chat1,
        },
        {
            "conversation_id": 2,
            "chat": chat2,
        },
    ])

    with patch("pandas.read_parquet", return_value=mock_df):
        yield


class TestConversationInfo:
    def test_creation(self) -> None:
        info = ConversationInfo(
            conversation_id=1,
            split="100K",
            message_count=100,
            estimated_tokens=50000,
            char_count=150000,
        )
        assert info.conversation_id == 1
        assert info.split == "100K"
        assert info.message_count == 100
        assert info.estimated_tokens == 50000


class TestLongContextDatasetInit:
    def test_defaults(self) -> None:
        ds = LongContextDataset()
        assert ds.dataset_name == "beam"
        assert ds.data_dir == _DEFAULT_DATA_DIR

    def test_custom(self) -> None:
        ds = LongContextDataset(dataset_name="beam", data_dir="/custom/path")
        assert ds.dataset_name == "beam"
        assert ds.data_dir == "/custom/path"

    def test_dataset_path(self) -> None:
        ds = LongContextDataset(data_dir="/test")
        assert ds.dataset_path == Path("/test/beam")

    def test_beam_data_path(self) -> None:
        ds = LongContextDataset(data_dir="/test")
        assert ds.beam_data_path == Path("/test/beam/data")


class TestLoadBeam:
    def test_load_100k(self, mock_beam_parquet, tmp_data_dir: str) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        records = ds.load(split="100K")
        assert len(records) == 2
        assert records[0]["conversation_id"] == 1
        assert records[1]["conversation_id"] == 2
        assert records[0]["split"] == "100K"

    def test_load_caches(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        r1 = ds.load(split="100K")
        r2 = ds.load(split="100K")
        assert r1 is r2

    def test_load_reloads_on_split_change(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        with pytest.raises((ValueError, FileNotFoundError)):
            ds.load(split="500K")

    def test_unknown_split(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        with pytest.raises(ValueError, match="Неизвестный split"):
            ds.load(split="unknown")

    def test_not_found(self) -> None:
        ds = LongContextDataset(data_dir="/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="не найден"):
            ds.load()

    def test_unknown_dataset(self, tmp_data_dir: str) -> None:
        ds = LongContextDataset(dataset_name="unknown", data_dir=tmp_data_dir)
        Path(tmp_data_dir, "unknown").mkdir(parents=True)
        with pytest.raises(ValueError, match="Неизвестный датасет"):
            ds.load()


class TestFlattenChat:
    def test_flatten_simple(self) -> None:
        chat = np.array([
            np.array([
                {"content": "Hi", "role": "user", "id": 0},
                {"content": "Hello", "role": "assistant", "id": 1},
            ]),
        ])
        result = LongContextDataset._flatten_chat(chat)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hi"}
        assert result[1] == {"role": "assistant", "content": "Hello"}

    def test_flatten_multiple_groups(self) -> None:
        chat = np.array([
            np.array([
                {"content": "msg1", "role": "user", "id": 0},
                {"content": "msg2", "role": "assistant", "id": 1},
            ]),
            np.array([
                {"content": "msg3", "role": "user", "id": 2},
                {"content": "msg4", "role": "assistant", "id": 3},
            ]),
        ])
        result = LongContextDataset._flatten_chat(chat)
        assert len(result) == 4
        assert [m["content"] for m in result] == ["msg1", "msg2", "msg3", "msg4"]

    def test_flatten_defaults_role(self) -> None:
        chat = np.array([
            np.array([
                {"content": "no role", "id": 0},
            ]),
        ])
        result = LongContextDataset._flatten_chat(chat)
        assert result[0]["role"] == "user"


class TestBuildInfo:
    def test_info_built_after_load(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        info = ds.get_conversations_info()
        assert len(info) == 2
        assert info[0].conversation_id == 1
        assert info[0].split == "100K"
        # First conversation has 4 messages (2 groups x 2)
        assert info[0].message_count == 4
        # Second has 2 messages
        assert info[1].message_count == 2


class TestGetFilteredConversations:
    def test_filter_by_tokens(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        # First conversation: ~40 chars / 3 = ~13 tokens
        # Second: ~6000 chars / 3 = ~2000 tokens
        filtered = ds.get_filtered_conversations(min_tokens=100, max_tokens=5000)
        assert len(filtered) == 1
        assert filtered[0].conversation_id == 2

    def test_filter_count_limit(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        filtered = ds.get_filtered_conversations(
            min_tokens=0, max_tokens=10000, count=1
        )
        assert len(filtered) == 1


class TestGetMessages:
    def test_basic_messages(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        msgs = ds.get_messages(conversation_id=0, max_tokens=100000)
        # system + 4 messages
        assert len(msgs) == 5
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello, can you help me?"

    def test_truncation(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        # Very small max_tokens -- should truncate
        msgs = ds.get_messages(conversation_id=1, max_tokens=5)
        # Should keep system + at least 1 message
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "system"

    def test_invalid_conversation_id(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        with pytest.raises(IndexError, match="вне диапазона"):
            ds.get_messages(conversation_id=99)

    def test_negative_conversation_id(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        with pytest.raises(IndexError):
            ds.get_messages(conversation_id=-1)


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert LongContextDataset._estimate_text_tokens("") == 0

    def test_short(self) -> None:
        assert LongContextDataset._estimate_text_tokens("hi") == 1

    def test_longer(self) -> None:
        est = LongContextDataset._estimate_text_tokens("Hello world! " * 100)
        assert est > 0
        assert est == len("Hello world! " * 100) // 3


class TestSummary:
    def test_not_available(self) -> None:
        ds = LongContextDataset(data_dir="/nonexistent")
        summary = ds.summary()
        assert summary["available"] is False
        assert summary["dataset"] == "beam"

    def test_available_with_data(
        self, mock_beam_parquet, tmp_data_dir: str
    ) -> None:
        ds = LongContextDataset(data_dir=tmp_data_dir)
        ds.load(split="100K")
        summary = ds.summary()
        assert summary["available"] is True
        assert summary["records"] == 2
        assert summary["split"] == "100K"
        assert "min_tokens" in summary
        assert "max_tokens" in summary
        assert "avg_tokens" in summary
        assert "total_messages" in summary
