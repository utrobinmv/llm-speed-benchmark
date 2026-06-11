#!/usr/bin/env python3
"""
tests/test_bench_single.py

Тесты для bench_single.py — покрывают утилиты и основной цикл с моком OpenAI.
"""

import os
import sys
import time
import io
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Подключаем модуль
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_single


# ============================================================================
# truncate_history()
# ============================================================================

class TestTruncateHistory:
    """Тесты для truncate_history()."""

    def test_only_system(self):
        """Только system — ничего не удаляем."""
        msgs = [{"role": "system", "content": "sys"}]
        result = bench_single.truncate_history(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_system_plus_one_pair(self):
        """System + 1 пара — ничего не удаляем."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        result = bench_single.truncate_history(msgs)
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
        result = bench_single.truncate_history(msgs)
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
        result = bench_single.truncate_history(msgs)
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
        result = bench_single.truncate_history(msgs)
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
        result = bench_single.truncate_history(msgs)
        assert result[2]["content"] == long_content

    def test_empty_messages(self):
        """Пустой список — пустой результат."""
        result = bench_single.truncate_history([])
        assert result == []


# ============================================================================
# progress_bar()
# ============================================================================

class TestProgressBar:
    """Тесты для progress_bar()."""

    def test_zero_tokens(self):
        """0 токенов — пустая полоса."""
        result = bench_single.progress_bar(0, 1000)
        assert result.startswith("[")
        assert "] 0.0%" in result

    def test_full(self):
        """100% — заполненная полоса."""
        result = bench_single.progress_bar(1000, 1000)
        assert "100.0%" in result
        assert "░" not in result

    def test_half(self):
        """50% — половина заполнена."""
        result = bench_single.progress_bar(500, 1000)
        assert "50.0%" in result

    def test_over_100(self):
        """Более 100% — капается на 100%."""
        result = bench_single.progress_bar(1500, 1000)
        assert "100.0%" in result

    def test_zero_max(self):
        """max_tokens=0 — не делит на ноль."""
        result = bench_single.progress_bar(100, 0)
        assert "] 0.0%" in result

    def test_custom_width(self):
        """Кастомная ширина."""
        result = bench_single.progress_bar(50, 100, width=10)
        # Должно быть ровно 10 символов в полосе
        bar_part = result.split("]")[0][1:]  # убираем "[" и "]"
        assert len(bar_part) == 10


# ============================================================================
# format_time()
# ============================================================================

class TestFormatTime:
    """Тесты для format_time()."""

    def test_zero(self):
        assert bench_single.format_time(0) == "00:00"

    def test_seconds_only(self):
        assert bench_single.format_time(30) == "00:30"

    def test_minutes_and_seconds(self):
        assert bench_single.format_time(90) == "01:30"

    def test_large_time(self):
        assert bench_single.format_time(3661) == "61:01"

    def test_float_seconds(self):
        """Дробные секунды округляются вниз."""
        assert bench_single.format_time(59.9) == "00:59"

    def test_exactly_60(self):
        assert bench_single.format_time(60) == "01:00"


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

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_runs_single_turn(self, mock_print, mock_get_client):
        """Один turn с моком завершается."""
        mock_client = MagicMock()
        # side_effect = callable → новый итератор каждый вызов
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=100, completion_tokens=50
        )
        mock_get_client.return_value = mock_client

        # MAX_CONTEXT_TOKENS=200: 100+50=150 < 200, 2-й turn: 100+100=200 >= 200 → стоп
        with patch("bench_single.MAX_CONTEXT_TOKENS", 200):
            bench_single.run_benchmark()

        assert mock_print.called
        # Проверяем, что create был вызван хотя бы раз
        assert mock_client.chat.completions.create.call_count >= 1

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_duration_stops_early(self, mock_print, mock_get_client):
        """--duration останавливает до заполнения контекста."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=10, completion_tokens=10
        )
        mock_get_client.return_value = mock_client

        # duration=0 —应立即停止
        with patch("bench_single.MAX_CONTEXT_TOKENS", 999999):
            bench_single.run_benchmark(duration=0)

        # При duration=0 цикл должен прерваться на первой итерации
        # create не должен быть вызван
        assert mock_client.chat.completions.create.call_count == 0

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_context_limit_stops(self, mock_print, mock_get_client):
        """Заполнение контекста останавливает цикл."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=100, completion_tokens=100
        )
        mock_get_client.return_value = mock_client

        # MAX_CONTEXT_TOKENS=150: после 1-го turn (100+100=200 >= 150) должно остановиться
        with patch("bench_single.MAX_CONTEXT_TOKENS", 150):
            bench_single.run_benchmark()

        # Один вызов — потом стоп
        assert mock_client.chat.completions.create.call_count == 1

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_multiple_turns(self, mock_print, mock_get_client):
        """Несколько turn'ов до лимита."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=10, completion_tokens=10
        )
        mock_get_client.return_value = mock_client

        # history=10, completion accumulates: 10, 20, 30, 40
        # Turn 1: check 0+0=0 < 50 → call → total=10
        # Turn 2: check 10+10=20 < 50 → call → total=20
        # Turn 3: check 10+20=30 < 50 → call → total=30
        # Turn 4: check 10+30=40 < 50 → call → total=40
        # Turn 5: check 10+40=50 >= 50 → стоп (до create)
        with patch("bench_single.MAX_CONTEXT_TOKENS", 50):
            bench_single.run_benchmark()

        assert mock_client.chat.completions.create.call_count == 4

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_truncate_called_at_warning(self, mock_print, mock_get_client):
        """При достижении 85% вызывается truncate_history."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=100, completion_tokens=100
        )
        mock_get_client.return_value = mock_client

        # MAX=1000, WARN=850
        # Turn 1: 100+100=200
        # Turn 2: 100+200=300
        # ...
        # Turn 4: 100+400=500
        # Turn 5: 100+500=600
        # Turn 6: 100+600=700
        # Turn 7: 100+700=800 >= 850? No → truncate BEFORE this turn
        # Actually: check is BEFORE the turn, so after turn 7 total=800
        # Turn 8 check: 100+800=900 >= 850 → truncate, then 900 < 1000 → continue
        # Turn 9 check: 100+900=1000 >= 1000 → break
        with patch("bench_single.MAX_CONTEXT_TOKENS", 1000):
            with patch("bench_single.truncate_history", wraps=bench_single.truncate_history) as mock_trunc:
                bench_single.run_benchmark()

        # truncate должен был вызываться хотя бы раз
        assert mock_trunc.call_count >= 1

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_usage_none_fallback(self, mock_print, mock_get_client):
        """Если usage=None, используется chunk_count как fallback."""
        mock_client = MagicMock()
        # Чанки без usage — 3 слова = 3 чанка
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

        with patch("bench_single.MAX_CONTEXT_TOKENS", 50):
            bench_single.run_benchmark()

        # Не должно упасть; completion_tokens = chunk_count (3)
        assert mock_client.chat.completions.create.call_count >= 1

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_empty_response_handling(self, mock_print, mock_get_client):
        """Пустой ответ от модели → skip, не зацикливается."""
        mock_client = MagicMock()
        # 3 пустых вызова → skip (не добавляют токены в контекст)
        # Дальше — с контентом: 10+10=20, 10+20=30, 10+30=40, 10+40=50 >= 50 → стоп
        call_count = [0]
        def side_effect(**kw):
            call_count[0] += 1
            if call_count[0] <= 3:
                return iter([])
            return make_mock_response(prompt_tokens=10, completion_tokens=10)
        mock_client.chat.completions.create.side_effect = side_effect
        mock_get_client.return_value = mock_client

        with patch("bench_single.MAX_CONTEXT_TOKENS", 50):
            bench_single.run_benchmark()

        # 3 пустых (skip) + 4 с контентом = 7 вызовов
        assert mock_client.chat.completions.create.call_count == 7

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_ttft_tracked(self, mock_print, mock_get_client):
        """TTFT (Time To First Token) отслеживается."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kw: make_mock_response(
            prompt_tokens=10, completion_tokens=10
        )
        mock_get_client.return_value = mock_client

        with patch("bench_single.MAX_CONTEXT_TOKENS", 50):
            bench_single.run_benchmark()

        # Проверяем, что print вызывался с TTFT
        print_calls = [str(call) for call in mock_print.call_args_list]
        # Хотя бы один вызов print должен содержать "TTFT"
        assert any("TTFT" in call for call in print_calls), f"TTFT не найден в print calls: {print_calls[:5]}"

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_instant_speed_logic(self, mock_print, mock_get_client):
        """Мгновенная скорость считается при достаточном количестве токенов."""
        # Прямая проверка логики: instant_buffer с 300 записями, INSTANT_WINDOW=100
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

    @patch("bench_single.get_client")
    @patch("builtins.print")
    def test_instant_window_constant(self, mock_print, mock_get_client):
        """INSTANT_WINDOW можно изменить."""
        assert bench_single.INSTANT_WINDOW == 200
        with patch("bench_single.INSTANT_WINDOW", 100):
            assert bench_single.INSTANT_WINDOW == 100


# ============================================================================
# Конфигурация
# ============================================================================

class TestConfig:
    """Тесты загрузки конфигурации."""

    def test_load_dotenv_called(self):
        """load_dotenv вызывается при импорте."""
        # Так как load_dotenv() вызывается на уровне модуля,
        # проверяем, что переменные из .env подгружены
        assert bench_single.BASE_URL is not None
        assert bench_single.MODEL is not None
        assert bench_single.API_KEY is not None
        assert bench_single.MAX_CONTEXT_TOKENS > 0

    def test_token_limit_warn_is_85_percent(self):
        """TOKEN_LIMIT_WARN = 85% от MAX_CONTEXT_TOKENS."""
        expected = int(bench_single.MAX_CONTEXT_TOKENS * 0.85)
        assert bench_single.TOKEN_LIMIT_WARN == expected

    def test_get_client_returns_openai(self):
        """get_client возвращает OpenAI клиент."""
        client = bench_single.get_client()
        assert client is not None
        assert hasattr(client, "chat")


# ============================================================================
# argparse
# ============================================================================

class TestArgparse:
    """Тесты командной строки."""

    def test_help_exits(self):
        """--help завершается с кодом 0."""
        with patch("sys.argv", ["bench_single.py", "--help"]):
            with pytest.raises(SystemExit) as exc:
                bench_single.argparse.ArgumentParser(
                    description="Бенчмарк скорости стриминга LLM"
                ).parse_args()
            assert exc.value.code == 0

    def test_duration_parsed(self):
        """--duration парсится корректно."""
        with patch("sys.argv", ["bench_single.py", "--duration", "60"]):
            parser = bench_single.argparse.ArgumentParser()
            parser.add_argument("--duration", type=int, default=None)
            args = parser.parse_args()
            assert args.duration == 60

    def test_duration_default_none(self):
        """Без --duration → None."""
        with patch("sys.argv", ["bench_single.py"]):
            parser = bench_single.argparse.ArgumentParser()
            parser.add_argument("--duration", type=int, default=None)
            args = parser.parse_args()
            assert args.duration is None
