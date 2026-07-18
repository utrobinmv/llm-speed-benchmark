"""
llm_speed_benchmark/long_context.py

Загрузка и подготовка датасетов длинных контекстов для бенчмарка.

Поддерживаемые датасеты:
- BEAM (Mohammadta/BEAM) -- multi-turn conversations 128K-10M токенов
- LongBench-Chat -- реальные разговоры 10K-100K токенов

Использование:
    from llm_speed_benchmark.long_context import LongContextDataset

    dataset = LongContextDataset(dataset_name="beam", cache_dir="~/.llm-speed-benchmark/data")
    dataset.download()

    # Получить длинные сообщения для воркера
    messages = dataset.get_messages(conversation_id=0, max_tokens=128000)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional


class LongContextDataset:
    """Загрузка и подготовка датасетов длинных контекстов.

    Args:
        dataset_name: Название датасета ("beam", "longbench-chat").
        cache_dir: Директория для кэширования скачанных данных.
        hf_token: HuggingFace token для доступа к gated датасетам.
    """

    def __init__(
        self,
        dataset_name: str = "beam",
        cache_dir: str = "~/.llm-speed-benchmark/data",
        hf_token: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.hf_token = hf_token
        self._data: Optional[list[dict]] = None

    @property
    def data_dir(self) -> Path:
        """Путь к директории с данными датасета."""
        return self.cache_dir / self.dataset_name

    @property
    def is_downloaded(self) -> bool:
        """Проверяет, скачан ли датасет."""
        return self.data_dir.exists() and any(self.data_dir.iterdir())

    def download(self) -> None:
        """Скачивает датасет из HuggingFace Hub.

        Использует `huggingface-cli download` для скачивания.
        Для gated датасетов требуется HF_TOKEN.
        """
        if self.is_downloaded:
            print(f"  Датасет {self.dataset_name} уже скачан: {self.data_dir}")
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset_name == "beam":
            self._download_beam()
        elif self.dataset_name == "longbench-chat":
            self._download_longbench_chat()
        else:
            raise ValueError(f"Неизвестный датасет: {self.dataset_name}. Доступные: beam, longbench-chat")

    def _download_beam(self) -> None:
        """Скачивает BEAM dataset с HuggingFace."""
        import subprocess

        repo_id = "Mohammadta/BEAM"
        cmd = [
            "huggingface-cli", "download",
            "--repo-type", "dataset",
            repo_id,
            "--local-dir", str(self.data_dir),
        ]
        if self.hf_token:
            cmd.extend(["--token", self.hf_token])

        print(f"  Скачивание BEAM dataset...")
        print(f"  {subprocess.list2cmdline(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Ошибка скачивания: {result.stderr}")
            raise RuntimeError(f"Не удалось скачать BEAM dataset: {result.stderr}")

        print(f"  BEAM dataset скачан в {self.data_dir}")

    def _download_longbench_chat(self) -> None:
        """Скачивает LongBench-Chat dataset."""
        import subprocess

        # LongBench-Chat доступен как часть LongBench
        repo_id = "gmlwns2000/LongBench-hip"
        cmd = [
            "huggingface-cli", "download",
            "--repo-type", "dataset",
            repo_id,
            "--local-dir", str(self.data_dir),
        ]
        if self.hf_token:
            cmd.extend(["--token", self.hf_token])

        print(f"  Скачивание LongBench-Chat dataset...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Ошибка скачивания: {result.stderr}")
            raise RuntimeError(f"Не удалось скачать LongBench-Chat: {result.stderr}")

        print(f"  LongBench-Chat скачан в {self.data_dir}")

    def load(self) -> list[dict]:
        """Загружает данные из локального кэша.

        Returns:
            Список словарей с данными датасета.
        """
        if self._data is not None:
            return self._data

        if not self.is_downloaded:
            raise FileNotFoundError(
                f"Датасет {self.dataset_name} не скачан. Вызовите download() или --download."
            )

        if self.dataset_name == "beam":
            self._data = self._load_beam()
        elif self.dataset_name == "longbench-chat":
            self._data = self._load_longbench_chat()
        else:
            raise ValueError(f"Неизвестный датасет: {self.dataset_name}")

        print(f"  Загружено {len(self._data)} записей из {self.dataset_name}")
        return self._data

    def _load_beam(self) -> list[dict]:
        """Загружает BEAM данные.

        BEAM содержит JSON файлы с conversations.
        Каждый файл -- одна conversation с метаданными.
        """
        records = []
        for json_file in sorted(self.data_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # BEAM может быть списком или словарём
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        return records

    def _load_longbench_chat(self) -> list[dict]:
        """Загружает LongBench-Chat данные."""
        records = []
        for json_file in sorted(self.data_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        return records

    def get_conversations(
        self,
        min_tokens: int = 50_000,
        max_tokens: int = 200_000,
        count: int = 10,
    ) -> list[dict]:
        """Фильтрует conversation по длине в токенах.

        Args:
            min_tokens: Минимальная длина conversation.
            max_tokens: Максимальная длина conversation.
            count: Максимальное количество conversation.

        Returns:
            Список conversation, отфильтрованных по длине.
        """
        data = self.load()
        filtered = []
        for record in data:
            tokens = self._estimate_tokens(record)
            if min_tokens <= tokens <= max_tokens:
                filtered.append({
                    "record": record,
                    "estimated_tokens": tokens,
                })
                if len(filtered) >= count:
                    break
        return filtered

    def get_messages(
        self,
        conversation_id: int = 0,
        max_tokens: int = 128_000,
    ) -> list[dict]:
        """Преобразует запись датасета в список сообщений для API.

        Args:
            conversation_id: Индекс conversation в датасете.
            max_tokens: Максимальное количество токенов (обрезка).

        Returns:
            Список сообщений в формате OpenAI API.
        """
        data = self.load()
        if conversation_id >= len(data):
            raise IndexError(
                f"conversation_id={conversation_id}超出范围 (всего {len(data)} записей)"
            )

        record = data[conversation_id]
        messages = self._record_to_messages(record)

        # Обрезка до max_tokens (приблизительно)
        if messages:
            total_est = sum(self._estimate_text_tokens(m.get("content", "")) for m in messages)
            while total_est > max_tokens and len(messages) > 2:
                messages.pop(1)  # Удаляем самые старые (после system)
                total_est = sum(
                    self._estimate_text_tokens(m.get("content", ""))
                    for m in messages
                )

        return messages

    def _record_to_messages(self, record: dict) -> list[dict]:
        """Преобразует запись датасета в формат сообщений OpenAI."""
        if self.dataset_name == "beam":
            return self._beam_to_messages(record)
        elif self.dataset_name == "longbench-chat":
            return self._longbench_to_messages(record)
        else:
            # Generic: ищем поле 'conversations' или 'messages'
            conversations = record.get("conversations") or record.get("messages") or []
            messages = [{"role": "system", "content": "Ты полезный помощник."}]
            for conv in conversations:
                if isinstance(conv, dict):
                    role = conv.get("role", "user")
                    content = conv.get("content", "")
                    if content:
                        messages.append({"role": role, "content": content})
                elif isinstance(conv, str):
                    messages.append({"role": "user", "content": conv})
            return messages

    def _beam_to_messages(self, record: dict) -> list[dict]:
        """BEAM -> OpenAI messages."""
        messages = [{"role": "system", "content": "Ты полезный помощник."}]

        # BEAM format: conversations = [{"from": "human"/"gpt", "value": "..."}]
        conversations = record.get("conversations") or record.get("dialogue") or []
        for conv in conversations:
            if isinstance(conv, dict):
                role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
                role = role_map.get(conv.get("from", ""), "user")
                content = conv.get("value") or conv.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        return messages

    def _longbench_to_messages(self, record: dict) -> list[dict]:
        """LongBench-Chat -> OpenAI messages."""
        messages = [{"role": "system", "content": "Ты полезный помощник."}]

        conversations = record.get("conversations") or record.get("messages") or []
        for conv in conversations:
            if isinstance(conv, dict):
                role = conv.get("role", "user")
                content = conv.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        return messages

    def _estimate_tokens(self, record: dict) -> int:
        """Оценивает количество токенов в записи."""
        conversations = record.get("conversations") or record.get("dialogue") or record.get("messages") or []
        total = 0
        for conv in conversations:
            if isinstance(conv, dict):
                text = conv.get("value") or conv.get("content", "")
            elif isinstance(conv, str):
                text = conv
            else:
                continue
            total += self._estimate_text_tokens(str(text))
        return total

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Примерная оценка токенов в тексте (1 токен ~ 4 символа для английского)."""
        if not text:
            return 0
        # Грубая оценка: ~4 chars per token для английского, ~1.5 для кириллицы
        # Используем 3.5 как среднее
        return max(1, len(text) // 3)

    def info(self) -> dict:
        """Возвращает информацию о датасете."""
        if not self.is_downloaded:
            return {
                "dataset": self.dataset_name,
                "downloaded": False,
                "cache_dir": str(self.data_dir),
            }

        data = self.load()
        if not data:
            return {
                "dataset": self.dataset_name,
                "downloaded": True,
                "records": 0,
                "cache_dir": str(self.data_dir),
            }

        # Статистика по длине
        lengths = [self._estimate_tokens(r) for r in data]
        return {
            "dataset": self.dataset_name,
            "downloaded": True,
            "records": len(data),
            "min_tokens": min(lengths) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
            "avg_tokens": sum(lengths) // len(lengths) if lengths else 0,
            "cache_dir": str(self.data_dir),
        }
