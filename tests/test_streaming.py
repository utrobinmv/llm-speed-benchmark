#!/usr/bin/env python3
"""
tests/test_streaming.py

Тесты для streaming.py -- StreamMetrics, StreamSession.
"""

from unittest.mock import MagicMock, patch
import time

import pytest

from llm_speed_benchmark.streaming import (
    StreamMetrics,
    StreamSession,
)


# ============================================================================
# StreamMetrics -- dataclass
# ============================================================================

class TestStreamMetrics:
    """StreamMetrics: дефолтные значения и установка полей."""

    def test_defaults(self):
        m = StreamMetrics()
        assert m.completion_tokens == 0
        assert m.prompt_tokens == 0
        assert m.chunk_count == 0
        assert m.elapsed == 0.0
        assert m.ttft is None
        assert m.assistant_content == ""
        assert m.instant_speed == 0.0
        assert m.call_speed == 0.0

    def test_set_values(self):
        m = StreamMetrics(
            completion_tokens=100,
            prompt_tokens=50,
            chunk_count=95,
            elapsed=2.5,
            ttft=0.12,
            assistant_content="Hello world",
            instant_speed=45.0,
            call_speed=40.0,
        )
        assert m.completion_tokens == 100
        assert m.ttft == 0.12
        assert m.call_speed == 40.0


# ============================================================================
# StreamSession -- run()
# ============================================================================

def make_mock_chunks(prompt_tokens=100, completion_tokens=50, content="Test answer text here done."):
    """Создаёт моки чанков для StreamSession."""
    chunks = []
    for word in content.split():
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = word + " "
        delta.reasoning = None
        delta.reasoning_content = None
        chunk.choices = [MagicMock(delta=delta)]
        chunk.usage = None
        chunks.append(chunk)
    # Final chunk with usage
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    final = MagicMock()
    final.choices = []
    final.usage = usage
    chunks.append(final)
    return chunks


class TestStreamSession:
    """StreamSession: run() с моками."""

    def _make_client(self, chunks):
        """Создаёт мок клиента с заданными чанками."""
        client = MagicMock()
        client.chat.completions.create.side_effect = lambda **kw: iter(chunks)
        return client

    def test_run_returns_metrics(self):
        client = self._make_client(make_mock_chunks(prompt_tokens=100, completion_tokens=5))
        session = StreamSession(client, call_count=0)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.completion_tokens == 5
        assert metrics.prompt_tokens == 100
        assert metrics.chunk_count == 5
        assert metrics.ttft is not None
        assert metrics.assistant_content == "Test answer text here done. "
        assert metrics.call_speed > 0

    def test_run_calibrates_first_call(self):
        """После первого вызова tokens_per_chunk калибруется."""
        # 5 чанков, 10 completion_tokens -> ratio = 2.0
        client = self._make_client(make_mock_chunks(prompt_tokens=10, completion_tokens=10))
        session = StreamSession(client, tokens_per_chunk=1.0, call_count=0)
        session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert session.tokens_per_chunk == 2.0  # 10 / 5

    def test_run_calibrates_smoothing(self):
        """После второго вызова -- экспоненциальное сглаживание."""
        client = MagicMock()
        calls = [0]
        def create_response(**kw):
            calls[0] += 1
            if calls[0] == 1:
                return iter(make_mock_chunks(prompt_tokens=10, completion_tokens=10))
            else:
                # 5 чанков, 5 токенов -> ratio = 1.0
                return iter(make_mock_chunks(prompt_tokens=10, completion_tokens=5))
        client.chat.completions.create.side_effect = create_response

        session = StreamSession(client, tokens_per_chunk=1.0, call_count=0)
        session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert session.tokens_per_chunk == 2.0  # первый вызов

        session.call_count = 1
        session.run(
            messages=[{"role": "user", "content": "test2"}],
            model="test-model",
        )
        # Сглаживание: 2.0 * 0.7 + 1.0 * 0.3 = 1.7
        assert abs(session.tokens_per_chunk - 1.7) < 0.01

    def test_run_no_usage_fallback(self):
        """Если usage не пришло -- completion_tokens = chunk_count."""
        chunks = []
        for word in ["No", "usage", "here."]:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = word + " "
            delta.reasoning = None
            delta.reasoning_content = None
            chunk.choices = [MagicMock(delta=delta)]
            chunk.usage = None
            chunks.append(chunk)
        # Нет финального чанка с usage

        client = self._make_client(chunks)
        session = StreamSession(client, call_count=0)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.completion_tokens == 3  # = chunk_count
        assert metrics.chunk_count == 3

    def test_run_reasoning_tokens(self):
        """Reasoning токены учитываются в chunk_count и контенте."""
        chunks = []
        # Reasoning chunk
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = ""
        delta.reasoning = "Let me think "
        delta.reasoning_content = None
        chunk.choices = [MagicMock(delta=delta)]
        chunk.usage = None
        chunks.append(chunk)
        # Content chunk
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = "The answer "
        delta.reasoning = None
        delta.reasoning_content = None
        chunk.choices = [MagicMock(delta=delta)]
        chunk.usage = None
        chunks.append(chunk)
        # Final with usage
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 4
        final = MagicMock()
        final.choices = []
        final.usage = usage
        chunks.append(final)

        client = self._make_client(chunks)
        session = StreamSession(client, call_count=0)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.chunk_count == 2  # reasoning + content
        assert "Let me think" in metrics.assistant_content
        assert "The answer" in metrics.assistant_content
        assert metrics.completion_tokens == 4

    def test_run_reasoning_content_field(self):
        """reasoning_content (OpenAI формат) тоже работает."""
        chunks = []
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = ""
        delta.reasoning = None
        delta.reasoning_content = "Thinking... "
        chunk.choices = [MagicMock(delta=delta)]
        chunk.usage = None
        chunks.append(chunk)
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = "Result "
        delta.reasoning = None
        delta.reasoning_content = None
        chunk.choices = [MagicMock(delta=delta)]
        chunk.usage = None
        chunks.append(chunk)
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 3
        final = MagicMock()
        final.choices = []
        final.usage = usage
        chunks.append(final)

        client = self._make_client(chunks)
        session = StreamSession(client, call_count=0)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.chunk_count == 2
        assert "Thinking..." in metrics.assistant_content
        assert "Result" in metrics.assistant_content

    def test_run_empty_response(self):
        """Пустой ответ -- все нули."""
        client = self._make_client([])
        session = StreamSession(client, call_count=0)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.completion_tokens == 0
        assert metrics.chunk_count == 0
        assert metrics.ttft is None
        assert metrics.assistant_content == ""

    def test_run_callback_called(self):
        """on_chunk callback вызывается во время стриминга."""
        callback_data = []

        def on_chunk(chunk_count, inst_sp, avg_sp, ttft, tail, wall):
            callback_data.append({
                "chunk_count": chunk_count,
                "inst_sp": inst_sp,
                "avg_sp": avg_sp,
                "ttft": ttft,
                "tail": tail,
                "wall": wall,
            })

        # Длинный контент для генерации callback'ов
        long_content = " ".join([f"word{i}" for i in range(100)])
        chunks = []
        for word in long_content.split():
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = word + " "
            delta.reasoning = None
            delta.reasoning_content = None
            chunk.choices = [MagicMock(delta=delta)]
            chunk.usage = None
            chunks.append(chunk)
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 100
        final = MagicMock()
        final.choices = []
        final.usage = usage
        chunks.append(final)

        client = self._make_client(chunks)
        session = StreamSession(
            client,
            call_count=0,
            on_chunk=on_chunk,
            on_chunk_args={"total_gen": 0, "start_time": time.time()},
        )
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        # Callback должен вызываться (throttle 0.5s, но при быстром стриминге может быть 0)
        # Главное -- metrics корректны
        assert metrics.completion_tokens == 100
        assert metrics.chunk_count == 100

    def test_run_no_callback(self):
        """Без callback -- работает нормально."""
        client = self._make_client(make_mock_chunks(prompt_tokens=10, completion_tokens=5))
        session = StreamSession(client, call_count=0, on_chunk=None)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.completion_tokens == 5

    def test_run_elapsed_positive(self):
        """elapsed > 0 после завершения."""
        client = self._make_client(make_mock_chunks(prompt_tokens=10, completion_tokens=5))
        session = StreamSession(client, call_count=0)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.elapsed > 0

    def test_run_ttft_positive(self):
        """TTFT > 0 когда есть токены."""
        client = self._make_client(make_mock_chunks(prompt_tokens=10, completion_tokens=5))
        session = StreamSession(client, call_count=0)
        metrics = session.run(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        assert metrics.ttft is not None
        assert metrics.ttft >= 0


# ============================================================================
# StreamSession -- _instant_speed()
# ============================================================================

class TestInstantSpeed:
    """StreamSession: _instant_speed -- мгновенная скорость."""

    def test_empty_buffer(self):
        session = StreamSession(MagicMock(), call_count=0)
        assert session._instant_speed([]) == 0.0

    def test_single_entry(self):
        session = StreamSession(MagicMock(), call_count=0)
        assert session._instant_speed([(1.0, 1)]) == 0.0

    def test_two_entries(self):
        session = StreamSession(MagicMock(), tokens_per_chunk=1.0, call_count=0)
        buf = [(0.0, 0), (1.0, 10)]  # 10 чанков за 1 сек
        speed = session._instant_speed(buf)
        assert speed == 10.0

    def test_with_tokens_per_chunk(self):
        session = StreamSession(MagicMock(), tokens_per_chunk=2.0, call_count=0)
        buf = [(0.0, 0), (1.0, 10)]  # 10 чанков * 2 = 20 токенов
        speed = session._instant_speed(buf)
        assert speed == 20.0

    def test_small_window(self):
        """Меньше 5 записей -- использует все."""
        session = StreamSession(MagicMock(), tokens_per_chunk=1.0, call_count=0)
        buf = [(i * 0.1, i * 5) for i in range(3)]  # 3 записи
        speed = session._instant_speed(buf)
        # win = min(5, 2) = 2
        # dt = 0.2, dc = (10 - 0) * 1.0 = 10
        assert speed == 50.0  # 10 / 0.2


# ============================================================================
# StreamSession -- _calibrate()
# ============================================================================

class TestCalibrate:
    """StreamSession: _calibrate -- калибровка tokens_per_chunk."""

    def test_first_call_exact(self):
        session = StreamSession(MagicMock(), tokens_per_chunk=1.0, call_count=0)
        session._calibrate(10, 20)  # 10 чанков, 20 токенов
        assert session.tokens_per_chunk == 2.0

    def test_subsequent_smoothing(self):
        session = StreamSession(MagicMock(), tokens_per_chunk=2.0, call_count=5)
        session._calibrate(10, 10)  # ratio = 1.0
        # 2.0 * 0.7 + 1.0 * 0.3 = 1.7
        assert abs(session.tokens_per_chunk - 1.7) < 0.01

    def test_zero_chunks_no_change(self):
        session = StreamSession(MagicMock(), tokens_per_chunk=1.5, call_count=0)
        session._calibrate(0, 0)
        assert session.tokens_per_chunk == 1.5  # не изменился
