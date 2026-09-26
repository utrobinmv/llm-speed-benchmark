"""
llm_speed_benchmark/transcription_utils.py

Утилиты для bench_transcription — бенчмарк ASR-моделей (Whisper, Qwen-ASR и др.).
Генерация аудио разной длины, измерение длительности, подготовка запросов.
"""

from __future__ import annotations

import io
import os
import struct
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Путь к бандлу аудиофайлов в репозитории
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "audio"

# Директория для кэша сгенерированных аудио
_CACHE_DIR = Path(os.path.expanduser("~/.llm-speed-benchmark/tmp/transcription"))

DEFAULT_TRANSCRIPTION_PROMPTS = [
    "Transcribe this audio.",
]


def get_audio_duration(audio_path: str | Path) -> float:
    """Получает длительность аудиофайла в секундах.

    Args:
        audio_path: Путь к аудиофайлу (.wav, .mp3).

    Returns:
        Длительность в секундах.
    """
    path = Path(audio_path)
    suffix = path.suffix.lower()

    if suffix == ".wav":
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    elif suffix == ".mp3":
        # Для MP3 используем размер файла как приблизительную оценку
        # или пытаемся использовать mutagen если доступен
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(path))
            return audio.info.length
        except ImportError:
            # Fallback: ~128kbps mono => ~16 bytes/ms
            size = path.stat().st_size
            return size / 16000.0
    else:
        # Fallback по размеру файла
        size = path.stat().st_size
        return size / 32000.0  # ~32kbps estimate


def get_audio_info(audio_path: str | Path) -> Dict[str, float]:
    """Получает информацию о аудиофайле.

    Args:
        audio_path: Путь к аудиофайлу.

    Returns:
        Словарь с duration, size_bytes, sample_rate, channels.
    """
    path = Path(audio_path)
    info: Dict[str, float] = {
        "duration": get_audio_duration(path),
        "size_bytes": float(path.stat().st_size),
    }

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            info["sample_rate"] = float(w.getframerate())
            info["channels"] = float(w.getnchannels())
            info["sampwidth"] = float(w.getsampwidth())
    else:
        info["sample_rate"] = 16000.0  # default estimate
        info["channels"] = 1.0
        info["sampwidth"] = 2.0

    return info


def load_bundled_audio() -> List[Path]:
    """Загружает пути к аудиофайлам из бандла."""
    return sorted(_ASSETS_DIR.glob("*.wav"))


def generate_audio_at_duration(
    target_duration: float,
    bundled_audios: Optional[List[Path]] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
) -> Path:
    """Генерирует аудиофайл указанной длительности путём зацикливания бандла.

    Результат кэшируется в ~/.llm-speed-benchmark/tmp/transcription/.

    Args:
        target_duration: Целевая длительность в секундах.
        bundled_audios: Список путей к исходным аудио (None = бандл).
        sample_rate: Частота дискретизации выходного файла.
        channels: Количество каналов.
        sampwidth: Ширина выборки в байтах.

    Returns:
        Путь к сгенерированному WAV файлу.
    """
    if bundled_audios is None:
        bundled_audios = load_bundled_audio()

    if not bundled_audios:
        raise FileNotFoundError("No bundled audio files found")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"audio_{int(target_duration * 100)}s.wav"

    if cache_file.exists():
        return cache_file

    # Считываем все бандл-аудио и циклически повторяем до нужной длины
    total_samples = int(target_duration * sample_rate)

    # Собираем raw frames из бандла
    source_frames: List[bytes] = []
    source_sample_rate = sample_rate

    for audio_path in bundled_audios:
        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            with wave.open(str(audio_path), "rb") as w:
                sr = w.getframerate()
                source_sample_rate = sr  # use last file's rate
                nframes = w.getnframes()
                raw = w.readframes(nframes)
                source_frames.append(raw)
        else:
            # Для не-WAV просто читаем raw bytes
            source_frames.append(audio_path.read_bytes())

    # Объединяем все бандл-фреймы в один пул
    pool = b"".join(source_frames)
    if not pool:
        raise ValueError("No audio data found in bundled files")

    # Рассчитываем длительность пула
    pool_duration = len(pool) / (sampwidth * sample_rate * channels)
    if pool_duration <= 0:
        raise ValueError("Audio pool duration is zero")

    # Циклически повторяем пул до нужной длины
    repeats_needed = int(total_samples * sampwidth * channels / len(pool)) + 1
    extended = pool * repeats_needed

    # Обрезаем до нужной длины
    target_bytes = total_samples * sampwidth * channels
    extended = extended[:target_bytes]

    # Записываем WAV
    with wave.open(str(cache_file), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframesraw(extended)

    return cache_file


def generate_duration_series(
    durations: List[float],
    bundled_audios: Optional[List[Path]] = None,
) -> List[Tuple[Path, float]]:
    """Генерирует серию аудиофайлов с указанными длительностями.

    Args:
        durations: Список целевых длительностей в секундах.
        bundled_audios: Список путей к исходным аудио (None = бандл).

    Returns:
        Список (path, actual_duration) кортежей.
    """
    results: List[Tuple[Path, float]] = []
    for dur in durations:
        path = generate_audio_at_duration(dur, bundled_audios)
        actual = get_audio_duration(path)
        results.append((path, actual))
    return results


def discover_audio_files(directory: str | Path) -> List[Path]:
    """Находит все аудио файлы в директории (.wav, .mp3, .ogg, .flac, .m4a).

    Args:
        directory: Директория для поиска.

    Returns:
        Отсортированный список Path к аудиофайлам.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    audio_files = sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])
    return audio_files


def filter_by_max_duration(
    audio_paths: List[Path],
    max_duration: Optional[float],
) -> List[Path]:
    """Фильтрует аудиофайлы по максимальной длительности.

    Args:
        audio_paths: Список путей к аудиофайлам.
        max_duration: Максимальная длительность в секундах (None = без фильтра).

    Returns:
        Отфильтрованный список путей.
    """
    if max_duration is None:
        return audio_paths

    result: List[Path] = []
    for p in audio_paths:
        dur = get_audio_duration(p)
        if dur <= max_duration:
            result.append(p)
    return result


def build_transcription_request(
    audio_path: str | Path,
    model: str,
    prompt: Optional[str] = None,
    language: Optional[str] = None,
    response_format: str = "json",
) -> Tuple[Path, Dict[str, str]]:
    """Подготавливает параметры для запроса audio.transcriptions.

    Args:
        audio_path: Путь к аудиофайлу.
        model: Название модели.
        prompt: Опциональный промпт для контекста.
        language: Опциональный язык (например, "ru").
        response_format: Формат ответа (json, text, srt, verbose_json, vtt).

    Returns:
        Кортеж (Path к файлу, dict с параметрами запроса).
    """
    params: Dict[str, str] = {
        "model": model,
        "response_format": response_format,
    }
    if prompt:
        params["prompt"] = prompt
    if language:
        params["language"] = language

    return Path(audio_path), params