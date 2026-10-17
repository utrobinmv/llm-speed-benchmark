"""
llm_speed_benchmark/audio_utils.py

Работа с тестовыми аудио для bench_audio.
Аудиофайлы хранятся в репозитории: assets/audio/
"""

from __future__ import annotations

import base64
import wave
from pathlib import Path
from typing import List, Sequence, Tuple

# Путь к бандлу аудиофайлов в репозитории
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "audio"

# Фразы, записанные в тестовые аудиофайлы (для справки)

DEFAULT_AUDIO_PROMPTS = [
    "Транскрибируй это аудио.",
    "Что сказано на этом аудио?",
    "Распознай текст на аудио.",
    "Перепиши то, что ты слышишь.",
    "Опиши содержание этого аудио.",
]


def get_bundled_audio_paths() -> List[Path]:
    """Возвращает пути ко всем аудиофайлам из бандла проекта."""
    return sorted(_ASSETS_DIR.glob("*.wav"))


def get_audio_paths(
    audio_dir: str | Path | None = None,
    count: int | None = None,
) -> List[Path]:
    """Возвращает пути к аудиофайлам для тестирования.

    Если audio_dir задан — ищет файлы там.
    Иначе — использует бандл из репозитория.

    Args:
        audio_dir: Директория с пользовательскими аудио (опционально).
        count: Лимит файлов (None = все доступные).

    Returns:
        Список Path к аудиофайлам.
    """
    if audio_dir:
        ap = Path(audio_dir)
        if not ap.is_dir():
            raise FileNotFoundError(
                f"audio_dir не существует или это не директория: {audio_dir}"
            )
        paths = discover_audio(audio_dir)
    else:
        paths = get_bundled_audio_paths()

    if count is not None:
        # count=0 явно запрашивает пустой список, count>0 — первые N файлов.
        # (None = без лимита, вернуть все.)
        paths = paths[:max(count, 0)]

    return paths


def load_audio_as_base64(audio_path: str | Path) -> Tuple[str, str]:
    """Загружает аудио и кодирует в base64.

    Args:
        audio_path: Путь к файлу аудио.

    Returns:
        Кортеж (format, base64_string).
    """
    suffix = Path(audio_path).suffix.lower()
    # OpenAI поддерживает только wav и mp3; неизвестный формат отклоняем,
    # а не молча маркируем как "wav" (иначе API получает неверный MIME).
    format_map = {
        ".wav": "wav",
        ".mp3": "mp3",
    }
    if suffix not in format_map:
        raise ValueError(
            f"Неподдерживаемый формат аудио '{suffix or '<без расширения>'}': {audio_path}. "
            "Поддерживаются только .wav и .mp3."
        )
    fmt = format_map[suffix]

    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return fmt, b64


def build_audio_message(
    audio_paths: Sequence[str | Path],
    prompt: str = "Транскрибируй это аудио.",
) -> list:
    """Создаёт message для audio-запроса (vLLM format).

    Использует data URL с base64 для аудио.

    Args:
        audio_paths: Путь или список путей к аудио.
        prompt: Текстовый промпт.

    Returns:
        Список сообщений в формате vLLM chat API.

    Raises:
        ValueError: если audio_paths пуст.
    """
    if not audio_paths:
        raise ValueError("build_audio_message: список аудиофайлов пуст")
    content_parts: list = [{"type": "text", "text": prompt}]
    for audio_path in audio_paths:
        fmt, b64 = load_audio_as_base64(audio_path)
        content_parts.append({
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/{fmt};base64,{b64}",
            },
        })
    return [{"role": "user", "content": content_parts}]


def discover_audio(directory: str | Path) -> List[Path]:
    """Находит все аудио файлы в директории.

    Args:
        directory: Директория для поиска.

    Returns:
        Отсортированный список Path к аудио (.wav, .mp3).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    extensions = {".wav", ".mp3"}
    audio_files = sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])
    return audio_files


def get_audio_info(audio_path: str | Path) -> dict:
    """Получает информацию о WAV файле.

    Args:
        audio_path: Путь к аудиофайлу.

    Returns:
        Словарь с channels, sample_rate, frames, duration.
    """
    path = Path(audio_path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            if rate <= 0:
                # Некорректный WAV (framerate 0): duration невычислим.
                # Не делим на 0 — возвращаем duration=None, чтобы вышестоящая
                # логика (фильтрация по длительности) отнеслась консервативно.
                return {
                    "channels": w.getnchannels(),
                    "sample_rate": rate,
                    "frames": w.getnframes(),
                    "duration": None,
                }
            return {
                "channels": w.getnchannels(),
                "sample_rate": rate,
                "frames": w.getnframes(),
                "duration": w.getnframes() / rate,
            }
    return {"size": path.stat().st_size}