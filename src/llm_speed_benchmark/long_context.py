"""
llm_speed_benchmark/long_context.py

Загрузка и подготовка датасетов длинных контекстов для бенчмарка.

Датасеты хранятся в ~/workspace/data/llm-speed-benchmark/datasets/
и должны быть скачаны заранее (см. DATA.md).

Поддерживаемые датасеты:
- BEAM (Mohammadta/BEAM) -- multi-turn conversations 128K-2.4M токенов

Использование:
    from llm_speed_benchmark.long_context import LongContextDataset

    dataset = LongContextDataset(
        dataset_name="beam",
        data_dir="/home/user/workspace/data/llm-speed-benchmark/datasets",
    )
    dataset.load(split="100K")

    # Получить сообщения для воркера (обрезка до 128K токенов)
    messages = dataset.get_messages(conversation_id=0, max_tokens=128_000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Default data directory
_DEFAULT_DATA_DIR = os.path.expanduser(
    "~/workspace/data/llm-speed-benchmark/datasets"
)

# BEAM splits and their parquet file patterns
BEAM_SPLITS = {
    "100K": "100K-*.parquet",
    "500K": "500K-*.parquet",
    "1M": "1M-*.parquet",
}


@dataclass
class ConversationInfo:
    """Информация о conversation из датасета."""

    conversation_id: int
    split: str
    message_count: int
    estimated_tokens: int
    char_count: int


@dataclass
class LongContextDataset:
    """Загрузка и подготовка датасетов длинных контекстов.

    Args:
        dataset_name: Название датасета ("beam").
        data_dir: Директория с заранее скачанными данными.
    """

    dataset_name: str = "beam"
    data_dir: str = _DEFAULT_DATA_DIR
    _conversations: Optional[list[dict]] = field(default=None, repr=False)
    _split: str = field(default="", repr=False)
    _info: list[ConversationInfo] = field(default_factory=list, repr=False)

    @property
    def dataset_path(self) -> Path:
        """Путь к директории с данными датасета."""
        return Path(self.data_dir) / self.dataset_name

    @property
    def beam_data_path(self) -> Path:
        """Путь к data/ внутри BEAM."""
        return self.dataset_path / "data"

    @property
    def is_available(self) -> bool:
        """Проверяет, есть ли данные на диске."""
        return self.dataset_path.exists()

    def load(self, split: str = "100K") -> list[dict]:
        """Загружает данные из локальной директории.

        Args:
            split: Сплит датасета ("100K", "500K", "1M").

        Returns:
            Список raw записей (dict с колонками parquet).

        Raises:
            FileNotFoundError: если данные не найдены.
            ValueError: если split неизвестен.
        """
        if self._conversations is not None and self._split == split:
            return self._conversations

        if not self.is_available:
            raise FileNotFoundError(
                f"Датасет {self.dataset_name} не найден: {self.dataset_path}\n"
                f"Скачайте заранее: hf download --repo-type dataset Mohammadta/BEAM --local-dir {self.dataset_path}"
            )

        if self.dataset_name == "beam":
            self._conversations = self._load_beam(split)
        else:
            raise ValueError(
                f"Неизвестный датасет: {self.dataset_name}. Доступные: beam"
            )

        self._split = split
        self._build_info()
        return self._conversations

    def _load_beam(self, split: str) -> list[dict]:
        """Загружает BEAM данные из parquet файлов.

        BEAM содержит parquet файлы по сплитам:
        - 100K: 20 conversations, ~133K-300K токенов
        - 500K: 35 conversations, ~430K-1.1M токенов
        - 1M: 35 conversations, ~947K-2.4M токенов

        Структура chat колонки: numpy array of numpy arrays,
        каждый внутренний массив -- группа сообщений (turn)
        с dict {content, role, id, ...}.
        """
        if split not in BEAM_SPLITS:
            raise ValueError(
                f"Неизвестный split '{split}' для BEAM. "
                f"Доступные: {', '.join(sorted(BEAM_SPLITS))}"
            )

        pattern = BEAM_SPLITS[split]
        parquet_files = sorted(self.beam_data_path.glob(pattern))
        if not parquet_files:
            raise FileNotFoundError(
                f"Не найдены parquet файлы для split={split}: "
                f"{self.beam_data_path}/{pattern}"
            )

        records = []
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            for idx in range(len(df)):
                row = df.iloc[idx]
                records.append({
                    "conversation_id": int(row["conversation_id"]),
                    "chat": row["chat"],  # numpy array of numpy arrays
                    "split": split,
                })

        return records

    def _build_info(self) -> None:
        """Собирает статистику по всем conversation."""
        assert self._conversations is not None
        self._info = []
        for record in self._conversations:
            msgs = self._flatten_chat(record["chat"])
            char_count = sum(len(m["content"]) for m in msgs)
            self._info.append(
                ConversationInfo(
                    conversation_id=record["conversation_id"],
                    split=record["split"],
                    message_count=len(msgs),
                    estimated_tokens=char_count // 3,
                    char_count=char_count,
                )
            )

    @staticmethod
    def _flatten_chat(chat_array) -> list[dict]:
        """Распаковывает chat (numpy array of numpy arrays) в плоский список.

        Каждый внутренний numpy array -- это группа сообщений (turn).
        Каждый элемент группы -- dict с полями content, role, id, ...
        """
        result: list[dict] = []
        for group in chat_array:
            for entry in group:
                if isinstance(entry, dict):
                    result.append({
                        "role": entry.get("role", "user"),
                        "content": entry.get("content", ""),
                    })
        return result

    def get_conversations_info(self) -> list[ConversationInfo]:
        """Возвращает информацию о всех загруженных conversation."""
        if self._info:
            return self._info
        self.load()
        return self._info

    def get_filtered_conversations(
        self,
        min_tokens: int = 50_000,
        max_tokens: int = 300_000,
        count: Optional[int] = None,
    ) -> list[ConversationInfo]:
        """Фильтрует conversation по длине в токенах.

        Args:
            min_tokens: Минимальная длина conversation.
            max_tokens: Максимальная длина conversation.
            count: Максимальное количество conversation. None = все.

        Returns:
            Список ConversationInfo, отфильтрованных по длине.
        """
        info_list = self.get_conversations_info()
        filtered = [
            ci for ci in info_list
            if min_tokens <= ci.estimated_tokens <= max_tokens
        ]
        if count is not None:
            filtered = filtered[:count]
        return filtered

    def get_messages(
        self,
        conversation_id: int = 0,
        max_tokens: int = 128_000,
    ) -> list[dict]:
        """Преобразует запись датасета в список сообщений для API.

        Args:
            conversation_id: Индекс conversation в загруженном сплите.
            max_tokens: Максимальное количество токенов (обрезка).

        Returns:
            Список сообщений в формате OpenAI API.

        Raises:
            IndexError: если conversation_id вне диапазона.
        """
        data = self.load()
        if conversation_id < 0 or conversation_id >= len(data):
            raise IndexError(
                f"conversation_id={conversation_id} вне диапазона "
                f"(всего {len(data)} записей в split={self._split})"
            )

        record = data[conversation_id]
        messages = self._beam_to_messages(record, max_tokens)
        return messages

    def _beam_to_messages(
        self, record: dict, max_tokens: int
    ) -> list[dict]:
        """BEAM -> OpenAI messages с обрезкой до max_tokens.

        Берёт сообщения из chat, добавляет system prompt,
        и обрезает до max_tokens, удаляя самые старые сообщения.
        """
        flat_msgs = self._flatten_chat(record["chat"])

        messages: list[dict] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        messages.extend(flat_msgs)

        # Обрезка до max_tokens -- удаляем самые старые (после system)
        total_est = sum(
            self._estimate_text_tokens(m.get("content", "")) for m in messages
        )
        while total_est > max_tokens and len(messages) > 2:
            messages.pop(1)  # Удаляем после system
            total_est = sum(
                self._estimate_text_tokens(m.get("content", ""))
                for m in messages
            )

        return messages

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Примерная оценка токенов в тексте."""
        if not text:
            return 0
        return max(1, len(text) // 3)

    def summary(self) -> dict:
        """Возвращает сводку по датасету."""
        if not self.is_available:
            return {
                "dataset": self.dataset_name,
                "available": False,
                "data_dir": str(self.dataset_path),
            }

        info_list = self.get_conversations_info()
        if not info_list:
            return {
                "dataset": self.dataset_name,
                "available": True,
                "records": 0,
                "data_dir": str(self.dataset_path),
            }

        tokens = [ci.estimated_tokens for ci in info_list]
        return {
            "dataset": self.dataset_name,
            "available": True,
            "split": self._split,
            "records": len(info_list),
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
            "avg_tokens": sum(tokens) // len(tokens),
            "total_messages": sum(ci.message_count for ci in info_list),
            "data_dir": str(self.dataset_path),
        }
