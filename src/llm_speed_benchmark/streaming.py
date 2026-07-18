"""
llm_speed_benchmark/streaming.py

Общая логика стриминга LLM -- извлечение из bench_single и bench_multi.

StreamMetrics  -- результат одного вызова (токены, TTFT, скорость, контент).
StreamSession  -- выполняет streaming-вызов, собирает метрики, калибрует tokens_per_chunk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from openai import OpenAI


@dataclass
class StreamMetrics:
    """Результаты одного streaming-вызова LLM."""

    completion_tokens: int = 0
    prompt_tokens: int = 0
    chunk_count: int = 0
    elapsed: float = 0.0
    ttft: Optional[float] = None
    assistant_content: str = ""
    # Мгновенная скорость (последние N чанков)
    instant_speed: float = 0.0
    # Скорость текущего вызова
    call_speed: float = 0.0


# Тип callback для live-обновлений во время стриминга.
# Вызывается каждые ~0.5s с текущими промежуточными данными.
ChunkCallback = Callable[
    [int, float, float, Optional[float], str, float],
    None,
]
# (chunk_count, instant_speed, avg_speed, ttft, content_tail, wall_elapsed)


class StreamSession:
    """Обёртка вокруг client.chat.completions.create(stream=True).

    Собирает метрики: токены, TTFT, мгновенную скорость, reasoning токены.
    Калибрует tokens_per_chunk после каждого вызова.

    Args:
        client: OpenAI-compatible клиент.
        tokens_per_chunk: Начальное соотношение (калибруется после 1-го вызова).
        call_count: Номер текущего вызова (для калибровки).
        on_chunk: Callback для live-обновлений (опционально).
        on_chunk_args: Дополнительные аргументы для callback (total_gen, start_time, worker_id).
    """

    def __init__(
        self,
        client: OpenAI,
        tokens_per_chunk: float = 1.0,
        call_count: int = 0,
        on_chunk: Optional[ChunkCallback] = None,
        on_chunk_args: Optional[dict] = None,
    ) -> None:
        self.client = client
        self.tokens_per_chunk = tokens_per_chunk
        self.call_count = call_count
        self.on_chunk = on_chunk
        self.on_chunk_args = on_chunk_args or {}

    def run(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> StreamMetrics:
        """Выполняет один streaming-вызов, возвращает метрики.

        Args:
            messages: Список сообщений (system + user/assistant пары).
            model: Название модели.
            temperature: Температура генерации.
            max_tokens: Максимум токенов в ответе.

        Returns:
            StreamMetrics с результатами вызова.
        """
        turn_start = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
        )

        completion_tokens = 0
        prompt_tokens = 0
        assistant_content = ""
        chunk_count = 0
        first_token_time = None

        # Для мгновенной скорости: (timestamp, cumulative_chunk)
        instant_buffer: list[tuple[float, int]] = []

        last_chunk_send = 0.0

        for chunk in response:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta is not None:
                    content = getattr(delta, "content", None) or ""
                    reasoning = getattr(delta, "reasoning", None) or ""
                    if not reasoning:
                        reasoning = getattr(delta, "reasoning_content", None) or ""

                    if content or reasoning:
                        if reasoning:
                            assistant_content += reasoning
                        if content:
                            assistant_content += content
                        chunk_count += 1
                        now_chunk = time.time()
                        if first_token_time is None:
                            first_token_time = now_chunk
                        instant_buffer.append((now_chunk, chunk_count))

                        # Live callback (throttle 0.5s)
                        if self.on_chunk and (now_chunk - last_chunk_send >= 0.5):
                            inst_sp = self._instant_speed(instant_buffer)
                            ttft = (
                                round(first_token_time - turn_start, 2)
                                if first_token_time is not None
                                else None
                            )
                            wall_elapsed = time.time() - self.on_chunk_args.get(
                                "start_time", turn_start
                            )
                            total_gen = self.on_chunk_args.get("total_gen", 0)
                            est_gen = total_gen + (chunk_count * self.tokens_per_chunk)
                            avg_sp = (
                                round(est_gen / wall_elapsed, 1)
                                if wall_elapsed > 0
                                else 0
                            )
                            self.on_chunk(
                                chunk_count,
                                inst_sp,
                                avg_sp,
                                ttft,
                                assistant_content[-100:],
                                wall_elapsed,
                            )
                            last_chunk_send = now_chunk

        elapsed = time.time() - turn_start

        # TTFT
        ttft = (first_token_time - turn_start) if first_token_time is not None else None

        # Мгновенная скорость
        inst_speed = self._instant_speed(instant_buffer)

        # Call speed
        call_speed = completion_tokens / elapsed if elapsed > 0 and completion_tokens > 0 else 0

        # Fallback: если usage не пришло
        if completion_tokens == 0:
            completion_tokens = chunk_count

        # Калибровка tokens_per_chunk
        self._calibrate(chunk_count, completion_tokens)

        return StreamMetrics(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            chunk_count=chunk_count,
            elapsed=elapsed,
            ttft=ttft,
            assistant_content=assistant_content,
            instant_speed=inst_speed,
            call_speed=call_speed,
        )

    def _instant_speed(self, buffer: list[tuple[float, int]]) -> float:
        """Мгновенная скорость на основе последних чанков из буфера."""
        if len(buffer) < 2:
            return 0.0
        win = min(5, len(buffer) - 1)
        t_start = buffer[-1 - win][0]
        t_end = buffer[-1][0]
        c_start = buffer[-1 - win][1]
        c_end = buffer[-1][1]
        dt = t_end - t_start
        dc = (c_end - c_start) * self.tokens_per_chunk
        return round(dc / dt, 1) if dt > 0 else 0.0

    def _calibrate(self, chunk_count: int, completion_tokens: int) -> None:
        """Калибрует tokens_per_chunk после завершения вызова."""
        if chunk_count > 0 and self.call_count == 0:
            # Первый вызов -- точное соотношение
            self.tokens_per_chunk = completion_tokens / chunk_count
        elif chunk_count > 0:
            # Экспоненциальное сглаживание
            self.tokens_per_chunk = (
                self.tokens_per_chunk * 0.7
                + (completion_tokens / chunk_count) * 0.3
            )
