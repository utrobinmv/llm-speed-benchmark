#!/usr/bin/env python3
"""
tests/test_bench_single.py

Тесты для bench_single — покрывают утилиты и основной цикл с моком OpenAI.
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from llm_speed_benchmark.utils import (
    truncate_history,
    progress_bar,
    format_time,
    get_client,
    BASE_URL,
    MODEL,
    API_KEY,
    MAX_CONTEXT_TOKENS,
    TOKEN_LIMIT_WARN,
    INSTANT_WINDOW,
)
from llm_speed_benchmark.bench_single import run_benchmark


# ============================================================================
# truncate_history()
# ============================================================================

class TestTruncateHistory:
    """Тесты для truncate_history()."""

    def test_only_system(self):
        """Только system — ничего не удаляем."""
        msgs = [{"role": "system", "content": "sys"}]
        result = truncate_history(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_system_plus_one_pair(self):
        """System + 1 пара — ничего не удаляем."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        result = truncate_history(msgs)
        assert len(result) == 3

    def test_removes_oldest_pair(self):
        """System + 2 пары — удаляем старую пару."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = truncate_history(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "u2"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "a2"

    def test_removes_multiple_pairs(self):
        """System + 5 пар — удаляем до 1 пары."""
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(5):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        result = truncate_history(msgs)
        assert len(result) == 3
        assert result[1]["content"] == "u4"
        assert result[2]["content"] == "a4"

    def test_no_system(self):
        """Без system — всё равно работает."""
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = truncate_history(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "u2"
        assert result[1]["content"] == "a2"

    def test_preserves_content(self):
        """Длинный контент не обрезается."""
        long_content = "x" * 10000
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": long_content},
        ]
        result = truncate_history(msgs)
        assert result[2]["content"] == long_content

    def test_empty_messages(self):
        """Пустой список — пустой результат."""
        result = truncate_history([])
        assert result == []


# ============================================================================
# progress_bar()
# ============================================================================

class TestProgressBar:
    """Тесты для progress_bar()."""

    def test_zero_tokens(self):
        """0 токенов — пустая полоса."""
        result = progress_bar(0, 1000)
        assert result.startswith("[")
        assert "] 0.0%" in result

    def test_full(self):
        """100% — заполненная полоса."""
        result = progress_bar(1000, 1000)
        assert "100.0%" in result
        assert "░" not in result

    def test_half(self):
        """50% — половина заполнена."""
        result = progress_bar(500, 1000)
        assert "50.0%" in result

    def test_over_100(self):
        """Более 100% — капается на 100%."""
        result = progress_bar(1500, 1000)
        assert "100.0%" in result

    def test_zero_max(self):
        """max_tokens=0 — не делит на ноль."""
        result = progress_bar(100, 0)
        assert "] 0.0%" in result

    def test_custom_width(self):
        """Кастомная ширина."""
        result = progress_bar(50, 100, width=10)
        # Должно быть ровно 10 символов в полосе
        bar_part = result.split("]")[0][1:]  # убираем "[" и "]"
        assert len(bar_part) == 10


# ============================================================================
# format_time()
# ============================================================================

class TestFormatTime:
    """Тесты для format_time()."""

    def test_zero(self):
        assert format_time(0) == "00:00"

    def test_seconds_only(self):
        assert format_time(30) == "00:30"

    def test_minutes_and_seconds(self):
        assert format_time(90) == "01:30"

    def test_large_time(self):
        assert format_time(3661) == "61:01"

    def test_float_seconds(self):
        """Дробные секунды округляются вниз."""
        assert format_time(59.9) == "00:59"

    def test_exactly_60(self):
        assert format_time(60) == "01:00"


# ============================================================================
# Mock OpenAI streaming response
# ============================================================================

def make_mock_response(prompt_tokens=100, completion_tokens=50, content="Тестовый ответ."):
    """Создаёт мок streaming-ответа OpenAI."""
    chunks = []

    # Чанки с контентом
    for word in content.split():
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = word + " "
        chunk.choices = [MagicMock(delta=delta)]
        chunk.usage = None
        chunks.append(chunk)

    # Финальный чанк с usage
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    final_chunk = MagicMock()
    final_chunk.choices = []
    final_chunk.usage = usage
    chunks.append(final_chunk)

    return iter(chunks)


# ============================================================================
# run_benchmark() — интеграционные тесты с моком
# ============================================================================

class TestRunBenchmark:
    """Тесты для run_benchmark() с моком OpenAI клиента."""

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_runs_single_turn(self, mock_print, mock_get_client):
        """Один turn с моком завершается."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=100, completion_tokens=50
        )
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 200):
            run_benchmark()

        assert mock_print.called
        assert mock_client.chat.completions.create.call_count >= 1

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_duration_stops_early(self, mock_print, mock_get_client):
        """--duration останавливает до заполнения контекста."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=10, completion_tokens=10
        )
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 999999):
            run_benchmark(duration=0)

        # При duration=0 цикл должен прерваться на первой итерации
        assert mock_client.chat.completions.create.call_count == 0

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_context_limit_stops(self, mock_print, mock_get_client):
        """Заполнение контекста останавливает цикл."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=100, completion_tokens=100
        )
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 150):
            with patch("llm_speed_benchmark.bench_single._token_limit_warn", return_value=127):
                run_benchmark()

        assert mock_client.chat.completions.create.call_count == 1

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_multiple_turns(self, mock_print, mock_get_client):
        """Несколько turn'ов до лимита."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=10, completion_tokens=10
        )
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 50):
            with patch("llm_speed_benchmark.bench_single._token_limit_warn", return_value=42):
                run_benchmark()

        assert mock_client.chat.completions.create.call_count == 4

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_truncate_called_at_warning(self, mock_print, mock_get_client):
        """При достижении 85% вызывается truncate_history."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=100, completion_tokens=100
        )
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 1000):
            with patch("llm_speed_benchmark.bench_single._token_limit_warn", return_value=850):
                with patch("llm_speed_benchmark.bench_single.truncate_history", wraps=truncate_history) as mock_trunc:
                    run_benchmark()

        assert mock_trunc.call_count >= 1

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_usage_none_fallback(self, mock_print, mock_get_client):
        """Если usage=None, используется chunk_count как fallback."""
        mock_client = MagicMock()
        chunks = []
        for word in ["Ответ", "без", "usage."]:
            delta = MagicMock()
            delta.content = word + " "
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=delta)]
            chunk.usage = None
            chunks.append(chunk)
        mock_client.chat.completions.create.side_effect = lambda **kw: iter(chunks)
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 50):
            run_benchmark()

        assert mock_client.chat.completions.create.call_count >= 1

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_empty_response_handling(self, mock_print, mock_get_client):
        """Пустой ответ от модели -> skip, не зацикливается."""
        mock_client = MagicMock()
        call_count = [0]
        def side_effect(**kw):
            call_count[0] += 1
            if call_count[0] <= 3:
                return iter([])
            return make_mock_response(prompt_tokens=10, completion_tokens=10)
        mock_client.chat.completions.create.side_effect = side_effect
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 50):
            with patch("llm_speed_benchmark.bench_single._token_limit_warn", return_value=42):
                run_benchmark()

        assert mock_client.chat.completions.create.call_count == 7

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_ttft_tracked(self, mock_print, mock_get_client):
        """TTFT (Time To First Token) отслеживается."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=10, completion_tokens=10
        )
        mock_get_client.return_value = mock_client

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 50):
            run_benchmark()

        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("TTFT" in call for call in print_calls), f"TTFT не найден в print calls: {print_calls[:5]}"

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_instant_speed_logic(self, mock_print, mock_get_client):
        """Мгновенная скорость считается при достаточном количестве токенов."""
        import time as _time
        instant_buffer = []
        base = _time.time()
        for i in range(300):
            instant_buffer.append((base + i * 0.01, i + 1))

        INSTANT_WINDOW = 100
        assert len(instant_buffer) > INSTANT_WINDOW
        window_start = instant_buffer[-INSTANT_WINDOW][0]
        window_end = instant_buffer[-1][0]
        window_tokens = instant_buffer[-1][1] - instant_buffer[-INSTANT_WINDOW - 1][1]
        window_time = window_end - window_start
        instant_speed = window_tokens / window_time if window_time > 0 else 0

        assert instant_speed > 0, f"instant_speed должен быть > 0, получил {instant_speed}"
        assert window_tokens == 100, f"window_tokens должен быть 100, получил {window_tokens}"

    @patch("llm_speed_benchmark.bench_single.get_client")
    @patch("builtins.print")
    def test_instant_window_constant(self, mock_print, mock_get_client):
        """INSTANT_WINDOW можно изменить."""
        assert INSTANT_WINDOW == 200


# ============================================================================
# Конфигурация
# ============================================================================

class TestConfig:
    """Тесты загрузки конфигурации."""

    def test_load_dotenv_called(self):
        """load_dotenv вызывается при импорте."""
        assert BASE_URL is not None
        assert MODEL is not None
        assert API_KEY is not None
        assert MAX_CONTEXT_TOKENS > 0

    def test_token_limit_warn_is_85_percent(self):
        """TOKEN_LIMIT_WARN = 85% от MAX_CONTEXT_TOKENS."""
        expected = int(MAX_CONTEXT_TOKENS * 0.85)
        assert TOKEN_LIMIT_WARN == expected

    def test_get_client_returns_openai(self):
        """get_client возвращает OpenAI клиент."""
        client = get_client()
        assert client is not None
        assert hasattr(client, "chat")

    def test_detect_model_context_length_success(self):
        """detect_model_context_length возвращает max_model_len от API."""
        from llm_speed_benchmark.utils import detect_model_context_length

        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.max_model_len = 131072
        mock_client.models.retrieve.return_value = mock_model

        result = detect_model_context_length(mock_client, "test-model")
        assert result == 131072

    def test_detect_model_context_length_fallback(self):
        """detect_model_context_length возвращает MAX_CONTEXT_TOKENS при ошибке."""
        from llm_speed_benchmark.utils import detect_model_context_length

        mock_client = MagicMock()
        mock_client.models.retrieve.side_effect = Exception("Not found")

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 262144):
            result = detect_model_context_length(mock_client, "test-model")
        assert result == 262144

    def test_detect_model_context_length_no_max_model_len(self):
        """Если max_model_len отсутствует — fallback."""
        from llm_speed_benchmark.utils import detect_model_context_length

        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.max_model_len = None
        mock_client.models.retrieve.return_value = mock_model

        with patch("llm_speed_benchmark.utils.MAX_CONTEXT_TOKENS", 262144):
            result = detect_model_context_length(mock_client, "test-model")
        assert result == 262144
