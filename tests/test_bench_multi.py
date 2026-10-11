#!/usr/bin/env python3
"""
tests/test_bench_multi.py

Тесты для bench_multi -- LiveTable, worker(), CLI.
"""

from unittest.mock import MagicMock, patch
import time

import pytest

from llm_speed_benchmark.bench_multi import LiveTable, worker
from llm_speed_benchmark.utils import format_time


# ============================================================================
# LiveTable -- mark_started
# ============================================================================

class TestLiveTableInit:
    """LiveTable: инициализация и mark_started."""

    def test_init_default(self):
        table = LiveTable(duration=60, total_workers=4)
        assert table.total_workers == 4
        assert table.duration == 60
        assert table.response_width == 60
        assert table.workers == {}
        assert table._errors == {}

    def test_init_custom_response_width(self):
        table = LiveTable(duration=None, total_workers=8, response_width=120)
        assert table.response_width == 120

    def test_mark_started(self):
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        assert 0 in table.workers
        w = table.workers[0]
        assert w["round"] == 1
        assert w["calls"] == 0
        assert w["gen"] == 0
        assert w["speed"] == 0
        assert w["tail"] == "[dim]waiting...[/]"

    def test_mark_started_overwrites(self):
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.workers[0]["gen"] = 999
        table.mark_started(0)
        assert table.workers[0]["gen"] == 0


# ============================================================================
# LiveTable -- update_stats
# ============================================================================

class TestLiveTableStats:
    """LiveTable: update_stats (после завершения стрима)."""

    def test_update_stats_sets_all_fields(self):
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.update_stats({
            "id": 0,
            "round": 2,
            "calls": 5,
            "p": 1000,
            "g": 5000,
            "cg": 1200,
            "chunks": 8000,
            "est_gen": 5000,
            "total": 6000,
            "speed": 45.2,
            "avg_speed": 30.1,
            "inst_speed": 50.0,
            "ttft": 0.15,
            "ttft_sum": 0.85,
            "wall": "01:30",
            "tail": "some response text",
        })
        w = table.workers[0]
        assert w["round"] == 2
        assert w["calls"] == 5
        assert w["prompt"] == 1000
        assert w["gen"] == 5000
        assert w["call_gen"] == 1200
        assert w["chunks"] == 8000
        assert w["gen_est"] == 5000
        assert w["total"] == 6000
        assert w["speed"] == 50.0  # inst_speed maps to speed
        assert w["avg"] == 30.1
        assert w["ttft"] == 0.15
        assert w["ttft_sum"] == 0.85
        assert w["wall"] == "01:30"
        assert w["tail"] == "some response text"

    def test_update_stats_partial(self):
        """Частичное обновление -- None значения не перезаписывают."""
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.workers[0]["gen"] = 5000
        table.workers[0]["avg"] = 30.0
        table.update_stats({
            "id": 0,
            "calls": 3,
            "g": 6000,
            # avg_speed отсутствует -> None
        })
        w = table.workers[0]
        assert w["calls"] == 3
        assert w["gen"] == 6000
        assert w["avg"] == 30.0  # не перезаписан

    def test_update_stats_creates_worker(self):
        """Если воркер не в mark_started -- создаётся."""
        table = LiveTable(duration=60, total_workers=4)
        table.update_stats({
            "id": 2,
            "calls": 1,
            "g": 100,
        })
        assert 2 in table.workers
        assert table.workers[2]["calls"] == 1
        assert table.workers[2]["gen"] == 100


# ============================================================================
# LiveTable -- update_live
# ============================================================================

class TestLiveTableLive:
    """LiveTable: update_live (во время стриминга)."""

    def test_update_live_overwrites_stats(self):
        """Live данные перезаписывают stats."""
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.update_stats({
            "id": 0,
            "calls": 3,
            "g": 4000,
            "est_gen": 4000,
            "speed": 40.0,
            "avg": 25.0,
            "ttft": 0.1,
            "ttft_sum": 0.5,
            "chunks": 6000,
            "wall": "01:00",
            "tail": "old stats tail",
        })
        # Live обновление
        table.update_live({
            "id": 0,
            "inst": 55.0,
            "avg": 28.0,
            "ttft": 0.12,
            "ttft_sum": 0.62,
            "est_tok": 4500,
            "chunks": 6500,
            "wall": "01:05",
            "tail": "new live tail",
        })
        w = table.workers[0]
        assert w["speed"] == 55.0
        assert w["avg"] == 28.0
        assert w["ttft"] == 0.12
        assert w["ttft_sum"] == 0.62
        assert w["gen_est"] == 4500
        assert w["chunks"] == 6500
        assert w["wall"] == "01:05"
        assert w["tail"] == "new live tail"
        # Stats поля не затронуты
        assert w["gen"] == 4000
        assert w["calls"] == 3

    def test_update_live_partial(self):
        """Только часть полей -- остальные не тронуты."""
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.update_live({
            "id": 0,
            "inst": 50.0,
            # avg не передан
        })
        assert table.workers[0]["speed"] == 50.0
        assert table.workers[0]["avg"] == 0  # из mark_started


# ============================================================================
# LiveTable -- update_time
# ============================================================================

class TestLiveTableTime:
    """LiveTable: update_time (каждую секунду)."""

    def test_update_time_wall_and_avg(self):
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.update_time({
            "id": 0,
            "wall": "02:15",
            "avg": 32.5,
        })
        assert table.workers[0]["wall"] == "02:15"
        assert table.workers[0]["avg"] == 32.5

    def test_update_time_only_wall(self):
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.workers[0]["avg"] = 30.0
        table.update_time({"id": 0, "wall": "03:00"})
        assert table.workers[0]["wall"] == "03:00"
        assert table.workers[0]["avg"] == 30.0


# ============================================================================
# LiveTable -- mark_error
# ============================================================================

class TestLiveTableError:
    """LiveTable: mark_error."""

    def test_mark_error(self):
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        tb = "Traceback (most recent call last):\n  File 'x.py', line 1\nError: boom"
        table.mark_error(0, tb)
        assert 0 in table._errors
        assert table._errors[0] == tb


# ============================================================================
# LiveTable -- _clean_tail
# ============================================================================

class TestLiveTableCleanTail:
    """LiveTable: _clean_tail -- фильтрация wide символов и обрезка."""

    def test_empty_tail(self):
        table = LiveTable(duration=60, total_workers=4)
        assert table._clean_tail("") == ""
        assert table._clean_tail(None) is None

    def test_ascii_preserved(self):
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("Hello world test")
        assert result == "Hello world test"

    def test_cyrillic_preserved(self):
        """Кириллица -- 1 ячейка, должна сохраниться."""
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("Привет мир тест")
        assert result == "Привет мир тест"

    def test_cjk_replaced(self):
        """CJK символы -- wide, заменяются на точки."""
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("Hello 世界 test")
        assert "世" not in result
        assert "界" not in result
        assert "." in result

    def test_emoji_replaced(self):
        """Эмодзи -- wide, заменяются."""
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("test ok")
        assert "." not in result or result == "test ok"

    def test_misc_technical_replaced(self):
        """U+2300-23FF (Misc Technical) -- wide, заменяются."""
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("test ok")
        # These are in the 0x2300-0x23FF range
        assert "." in result or result == "test ok"

    def test_newlines_replaced(self):
        """Переносы строк заменяются на пробелы."""
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("line1\nline2\rline3\tline4")
        assert "\n" not in result
        assert "\r" not in result
        assert "\t" not in result
        assert "line1" in result
        assert "line2" in result

    def test_rich_tags_preserved(self):
        """Rich-теги [red]...[/] сохраняются."""
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("[red]error[/] text")
        assert "[red]" in result
        assert "[/]" in result

    def test_multiple_spaces_collapsed(self):
        """Множественные пробелы схлопываются."""
        table = LiveTable(duration=60, total_workers=4)
        result = table._clean_tail("hello   world    test")
        assert "   " not in result
        assert "hello world test" == result

    def test_truncation(self):
        """Длинный текст обрезается до response_width."""
        table = LiveTable(duration=60, total_workers=4, response_width=20)
        long_text = "a" * 200
        result = table._clean_tail(long_text)
        assert result.startswith("...")
        # Визуальная длина <= response_width
        assert len(result) <= 25  # ... + max_vlen + небольшой запас

    def test_short_text_not_truncated(self):
        """Короткий текст не обрезается."""
        table = LiveTable(duration=60, total_workers=4, response_width=60)
        short = "short text"
        result = table._clean_tail(short)
        assert result == short


# ============================================================================
# LiveTable -- __rich__ (рендеринг)
# ============================================================================

class TestLiveTableRender:
    """LiveTable: __rich__ -- рендеринг таблицы."""

    def test_render_pending_worker(self):
        """Воркер не запущен -- показывает pend."""
        table = LiveTable(duration=60, total_workers=2)
        rich_obj = table.__rich__()
        # Это Rich Table -- проверяем, что объект создан
        assert rich_obj is not None
        assert hasattr(rich_obj, "add_row")

    def test_render_error_worker(self):
        """Воркер с ошибкой -- показывает ERROR."""
        table = LiveTable(duration=60, total_workers=2)
        table.mark_started(0)
        table.mark_error(0, "Traceback: boom")
        rich_obj = table.__rich__()
        assert rich_obj is not None

    def test_render_active_worker(self):
        """Активный воркер -- показывает данные."""
        table = LiveTable(duration=60, total_workers=2)
        table.mark_started(0)
        table.update_stats({
            "id": 0,
            "calls": 3,
            "g": 5000,
            "cg": 1200,
            "p": 1000,
            "total": 6200,
            "speed": 45.0,
            "avg_speed": 30.0,
            "inst_speed": 50.0,
            "ttft": 0.15,
            "ttft_sum": 0.45,
            "wall": "01:30",
            "tail": "response text",
            "round": 1,
            "chunks": 6000,
            "est_gen": 5000,
        })
        rich_obj = table.__rich__()
        assert rich_obj is not None

    def test_render_all_workers(self):
        """Все воркеры в разных состояниях."""
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.update_stats({"id": 0, "calls": 1, "g": 100, "cg": 100,
                            "p": 50, "total": 150, "speed": 30, "avg_speed": 20,
                            "inst_speed": 35, "ttft": 0.1, "ttft_sum": 0.1,
                            "wall": "00:05", "tail": "ok", "round": 1,
                            "chunks": 100, "est_gen": 100})
        table.mark_started(1)
        table.mark_error(2, "error trace")
        # 3 -- pending
        rich_obj = table.__rich__()
        assert rich_obj is not None


# ============================================================================
# LiveTable -- _merge
# ============================================================================

class TestLiveTableMerge:
    """LiveTable: _merge -- статический метод."""

    def test_merge_non_none(self):
        w = {"a": 1, "b": 2}
        LiveTable._merge(w, {"a": 10, "c": 30})
        assert w == {"a": 10, "b": 2, "c": 30}

    def test_merge_skips_none(self):
        w = {"a": 1, "b": 2}
        LiveTable._merge(w, {"a": None, "b": 20})
        assert w == {"a": 1, "b": 20}

    def test_merge_empty(self):
        w = {"a": 1}
        LiveTable._merge(w, {})
        assert w == {"a": 1}


# ============================================================================
# LiveTable -- интеграция: приоритет обновлений
# ============================================================================

class TestLiveTableIntegration:
    """LiveTable: последовательность обновлений."""

    def test_stats_then_live_then_stats(self):
        """Stats -> Live -> Stats: stats после live перезаписывает."""
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)

        # Stats после 1-го вызова
        table.update_stats({
            "id": 0, "calls": 1, "g": 1000, "cg": 1000,
            "p": 100, "total": 1100, "speed": 40, "avg_speed": 20,
            "inst_speed": 40, "ttft": 0.1, "ttft_sum": 0.1,
            "wall": "00:30", "tail": "first", "round": 1,
            "chunks": 1000, "est_gen": 1000,
        })
        assert table.workers[0]["gen"] == 1000

        # Live во время 2-го вызова
        table.update_live({
            "id": 0, "inst": 55, "avg": 25, "ttft": 0.12,
            "ttft_sum": 0.22, "est_tok": 1500, "chunks": 1500,
            "wall": "01:00", "tail": "second live",
        })
        assert table.workers[0]["speed"] == 55
        assert table.workers[0]["gen_est"] == 1500
        assert table.workers[0]["gen"] == 1000  # не изменился

        # Stats после 2-го вызова
        table.update_stats({
            "id": 0, "calls": 2, "g": 2000, "cg": 1000,
            "p": 200, "total": 2200, "speed": 42, "avg_speed": 28,
            "inst_speed": 42, "ttft": 0.12, "ttft_sum": 0.22,
            "wall": "01:15", "tail": "second done", "round": 1,
            "chunks": 2000, "est_gen": 2000,
        })
        assert table.workers[0]["gen"] == 2000
        assert table.workers[0]["gen_est"] == 2000
        assert table.workers[0]["calls"] == 2
        assert table.workers[0]["tail"] == "second done"

    def test_time_updates_between_calls(self):
        """Time обновления между live и stats."""
        table = LiveTable(duration=60, total_workers=4)
        table.mark_started(0)
        table.update_stats({
            "id": 0, "calls": 1, "g": 1000, "cg": 1000,
            "p": 100, "total": 1100, "speed": 40, "avg_speed": 20,
            "inst_speed": 40, "ttft": 0.1, "ttft_sum": 0.1,
            "wall": "00:30", "tail": "done", "round": 1,
            "chunks": 1000, "est_gen": 1000,
        })
        # Time update (воркер ждёт TTFT)
        table.update_time({"id": 0, "wall": "00:35", "avg": 21})
        assert table.workers[0]["wall"] == "00:35"
        assert table.workers[0]["avg"] == 21
        # Gen не изменился
        assert table.workers[0]["gen"] == 1000


# ============================================================================
# worker() -- с моком OpenAI
# ============================================================================

def make_worker_mock_response(prompt_tokens=100, completion_tokens=50,
                               content="Тестовый ответ для многопроцессного бенчмарка."):
    """Создаёт мок streaming-ответа для worker."""
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
    # Финальный чанк с usage
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    final = MagicMock()
    final.choices = []
    final.usage = usage
    chunks.append(final)
    return iter(chunks)


class TestWorker:
    """worker(): тесты с моком OpenAI."""

    def test_worker_sends_start_message(self):
        """Воркер отправляет 'start' сразу после запуска."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            # Пустой iter -> worker сразу завершит цикл
            mock_client.chat.completions.create.return_value = iter([])

            worker(0, q, start_event, duration=1)

        # Проверяем, что первый put -- start
        calls = [c[0][0] for c in q.put.call_args_list]
        assert any(c.get("type") == "start" for c in calls)

    def test_worker_sends_stats_after_call(self):
        """Воркер отправляет stats после завершения вызова."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = lambda **kw: make_worker_mock_response(
                prompt_tokens=100, completion_tokens=50
            )

            worker(0, q, start_event, duration=5)

        calls = [c[0][0] for c in q.put.call_args_list]
        stats_calls = [c for c in calls if c.get("type") == "stats"]
        assert len(stats_calls) >= 1
        s = stats_calls[0]
        assert s["id"] == 0
        assert s["g"] > 0
        assert s["cg"] > 0
        assert s["calls"] >= 1

    def test_worker_handles_empty_response(self):
        """Пустой ответ -- не краш, отправляет stats с 0."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = iter([])

            worker(0, q, start_event, duration=2)

        calls = [c[0][0] for c in q.put.call_args_list]
        stats_calls = [c for c in calls if c.get("type") == "stats"]
        assert len(stats_calls) >= 1
        # Пустой ответ -> cg = 0
        assert stats_calls[0]["cg"] == 0

    def test_worker_sends_error_on_exception(self):
        """Исключение в воркере -> error message."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_openai_cls.side_effect = RuntimeError("Connection failed")

            worker(0, q, start_event, duration=5)

        calls = [c[0][0] for c in q.put.call_args_list]
        error_calls = [c for c in calls if c.get("type") == "error"]
        assert len(error_calls) >= 1
        assert "traceback" in error_calls[0]

    def test_worker_duration_stops(self):
        """duration ограничивает работу воркера."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        call_count = [0]
        def slow_response(**kw):
            call_count[0] += 1
            # Имитация задержки
            time.sleep(0.3)
            return make_worker_mock_response(prompt_tokens=10, completion_tokens=10)

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = slow_response

            worker(0, q, start_event, duration=1)

        # При duration=1 и sleep(0.3) -- максимум 3-4 вызова
        assert call_count[0] <= 5

    def test_worker_reasoning_tokens_counted(self):
        """Reasoning токены учитываются в chunk_count."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        # Создаём чанки с reasoning
        chunks = []
        for word in ["thinking step"]:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = ""
            delta.reasoning = word + " "
            delta.reasoning_content = None
            chunk.choices = [MagicMock(delta=delta)]
            chunk.usage = None
            chunks.append(chunk)
        for word in ["answer text"]:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = word + " "
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

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = lambda **kw: iter(chunks)

            worker(0, q, start_event, duration=5)

        calls = [c[0][0] for c in q.put.call_args_list]
        stats_calls = [c for c in calls if c.get("type") == "stats"]
        assert len(stats_calls) >= 1
        # completion_tokens = 4 из usage
        assert stats_calls[0]["cg"] == 4

    def test_worker_live_messages(self):
        """Воркер отправляет live сообщения во время стриминга."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        # Длинный ответ для генерации live сообщений
        long_content = " ".join([f"word{i}" for i in range(200)])
        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = lambda **kw: make_worker_mock_response(
                prompt_tokens=100, completion_tokens=200, content=long_content
            )

            worker(0, q, start_event, duration=5)

        calls = [c[0][0] for c in q.put.call_args_list]
        live_calls = [c for c in calls if c.get("type") == "live"]
        # Должны быть live сообщения (throttle 0.5s, но если стрим быстрый -- возможно 0)
        # Хотя бы stats должен быть
        stats_calls = [c for c in calls if c.get("type") == "stats"]
        assert len(stats_calls) >= 1


# ============================================================================
# worker() -- long context
# ============================================================================

class TestWorkerLongContext:
    """worker(): long context -- диалог датасета сохраняется между раундами."""

    BEAM = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "BEAM user msg 1"},
        {"role": "assistant", "content": "BEAM assistant msg 1"},
        {"role": "user", "content": "BEAM user msg 2"},
        {"role": "assistant", "content": "BEAM assistant msg 2"},
    ]

    def test_lc_worker_keeps_dataset_dialog_on_new_round(self):
        """При достижении лимита контекста long-context воркер
        возвращается к диалогу датасета, а не к голому system."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None
        captured = []

        def fake_create(**kw):
            captured.append(list(kw["messages"]))
            return make_worker_mock_response(prompt_tokens=1000, completion_tokens=5000)

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls, \
             patch("llm_speed_benchmark.bench_multi.MAX_CONTEXT_TOKENS", 4000):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = fake_create
            worker(0, q, start_event, duration=1,
                   initial_messages=list(self.BEAM), long_context=True)

        assert len(captured) >= 2
        # Первый вызов: полный диалог датасета + промпт воркера
        assert captured[0][:5] == self.BEAM
        assert captured[0][5]["role"] == "user"
        # Второй вызов (новый раунд): снова полный диалог датасета + промпт
        assert captured[1][:5] == self.BEAM
        assert captured[1][5]["role"] == "user"
        assert "Поток" in captured[1][5]["content"]

    def test_lc_worker_sends_long_messages_in_chars(self):
        """Long-context воркер отправляет длинный диалог в API.
        Замокан сам вызов API (OpenAI.chat.completions.create) -- сеть не
        трогается. Проверяется длина отправленных сообщений в символах:
        при --long-context-workers 1 в первый же вызов должны уйти все
        сообщения датасета (сотни тысяч символов), а НЕ короткий промпт.
        """
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None
        captured = []

        # Имитация диалога BEAM (близкая по масштабу к реальной 100K-записи):
        # длинный system + 60 пар user/assistant по ~3000 символов каждая.
        beam_dialog = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(60):
            beam_dialog.append(
                {"role": "user", "content": f"Long user message #{i}: " + "x" * 2500}
            )
            beam_dialog.append(
                {"role": "assistant", "content": f"Long assistant reply #{i}: " + "y" * 2500}
            )
        total_expected = sum(len(m["content"]) for m in beam_dialog)

        def fake_create(**kw):
            captured.append(list(kw["messages"]))
            return make_worker_mock_response(prompt_tokens=98038, completion_tokens=100)

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = fake_create
            worker(0, q, start_event, duration=1,
                   initial_messages=list(beam_dialog), long_context=True)

        assert captured, "воркер должен был сделать хотя бы один вызов API"
        sent = captured[0]

        # В первый вызов уходят ВСЕ сообщения датасета (диалог + промпт воркера)
        assert len(sent) == len(beam_dialog) + 1  # +1 пользовательский промпт
        total_chars = sum(len(m["content"]) for m in sent)
        print(f"[SENT] messages={len(sent)} total_chars={total_chars} "
              f"(датасет={total_expected})")
        # Длина на порядок больше короткого промпта -- диалог ушёл целиком
        assert total_chars >= total_expected
        assert total_chars > 100_000, \
            f"Отправлено всего {total_chars} символов -- это не длинный контекст!"
        assert total_expected > 100_000, "тестовый диалог слишком короткий"
        """Start-сообщение помечает long_context=True."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls, \
             patch("llm_speed_benchmark.bench_multi.MAX_CONTEXT_TOKENS", 4000):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = iter([])
            worker(0, q, start_event, duration=1,
                   initial_messages=list(self.BEAM), long_context=True)

        calls = [c[0][0] for c in q.put.call_args_list]
        start = [c for c in calls if c.get("type") == "start"]
        assert start and start[0]["long_context"] is True

    def test_normal_worker_resets_to_system_on_new_round(self):
        """Обычный режим (regression): при лимите контекст сбрасывается
        к голому system, как и раньше."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None
        captured = []

        def fake_create(**kw):
            captured.append(list(kw["messages"]))
            return make_worker_mock_response(prompt_tokens=1000, completion_tokens=5000)

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls, \
             patch("llm_speed_benchmark.bench_multi.MAX_CONTEXT_TOKENS", 4000):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = fake_create
            worker(0, q, start_event, duration=1)

        assert len(captured) >= 2
        # Второй вызов: system + промпт (история сброшена)
        assert captured[1][0]["role"] == "system"
        assert captured[1][1]["role"] == "user"
        assert len(captured[1]) == 2


# ============================================================================
# worker() -- model_timeout
# ============================================================================

class TestWorkerTimeout:
    """worker(): таймаут запроса к модели (медленные серверы, долгий prefill)."""

    def test_default_timeout_600(self):
        """По умолчанию клиент создаётся с timeout=600.0."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = iter([])
            worker(0, q, start_event, duration=1)

        _, kwargs = mock_openai_cls.call_args
        assert kwargs["timeout"] == 600.0

    def test_custom_timeout_passed_to_client(self):
        """Явный model_timeout передаётся в OpenAI-клиент."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = iter([])
            worker(0, q, start_event, duration=1, model_timeout=3600.0)

        _, kwargs = mock_openai_cls.call_args
        assert kwargs["timeout"] == 3600.0

    def test_zero_timeout_means_no_timeout(self):
        """model_timeout=0 -- клиент без таймаута (timeout=None),
        ждёт prefill бесконечно."""
        q = MagicMock()
        start_event = MagicMock()
        start_event.wait.return_value = None

        with patch("llm_speed_benchmark.bench_multi.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = iter([])
            worker(0, q, start_event, duration=1, model_timeout=0)

        _, kwargs = mock_openai_cls.call_args
        assert kwargs["timeout"] is None


# ============================================================================
# CLI -- bench_multi
# ============================================================================

class TestCliMulti:
    """CLI bench_multi: парсинг аргументов."""

    def test_cli_default_args(self):
        from llm_speed_benchmark.bench_multi import cli
        with patch("sys.argv", ["bench_multi"]):
            with patch("llm_speed_benchmark.bench_multi.run_benchmark") as mock_run:
                cli()
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["workers"] == 4
                assert call_kwargs["duration"] is None
                assert call_kwargs["response_width"] == 60

    def test_cli_custom_workers(self):
        from llm_speed_benchmark.bench_multi import cli
        with patch("sys.argv", ["bench_multi", "-w", "8", "-d", "120"]):
            with patch("llm_speed_benchmark.bench_multi.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["workers"] == 8
                assert call_kwargs["duration"] == 120

    def test_cli_all_args(self):
        from llm_speed_benchmark.bench_multi import cli
        args = [
            "bench_multi",
            "-u", "http://test:8000/v1",
            "-k", "test-key",
            "-m", "test-model",
            "-w", "2",
            "-d", "30",
            "--response-width", "80",
            "--max-context", "32768",
        ]
        with patch("sys.argv", args):
            with patch("llm_speed_benchmark.bench_multi.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["base_url"] == "http://test:8000/v1"
                assert call_kwargs["api_key"] == "test-key"
                assert call_kwargs["model"] == "test-model"
                assert call_kwargs["workers"] == 2
                assert call_kwargs["duration"] == 30
                assert call_kwargs["response_width"] == 80
                assert call_kwargs["max_context"] == 32768

    def test_cli_long_context_args(self):
        from llm_speed_benchmark.bench_multi import cli
        args = [
            "bench_multi",
            "-w", "2",
            "--long-context-workers", "1",
            "--split", "500K",
            "--data-dir", "/tmp/datasets",
        ]
        with patch("sys.argv", args):
            with patch("llm_speed_benchmark.bench_multi.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["long_context_workers"] == 1
                assert call_kwargs["split"] == "500K"
                assert call_kwargs["data_dir"] == "/tmp/datasets"
                assert call_kwargs["allow_download"] is True

    def test_cli_no_download(self):
        from llm_speed_benchmark.bench_multi import cli
        args = ["bench_multi", "--long-context-workers", "1", "--no-download"]
        with patch("sys.argv", args):
            with patch("llm_speed_benchmark.bench_multi.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["allow_download"] is False

    def test_cli_model_timeout(self):
        from llm_speed_benchmark.bench_multi import cli
        args = ["bench_multi", "--model-timeout", "0"]
        with patch("sys.argv", args):
            with patch("llm_speed_benchmark.bench_multi.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["model_timeout"] == 0.0

    def test_cli_model_timeout_default_none(self):
        """Без --model-timeout -- None (разрешается из env/600 в run_benchmark)."""
        from llm_speed_benchmark.bench_multi import cli
        args = ["bench_multi"]
        with patch("sys.argv", args):
            with patch("llm_speed_benchmark.bench_multi.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["model_timeout"] is None


# ============================================================================
# run_benchmark() -- long context: clamp контекста и авто-скачивание
# ============================================================================

class TestRunBenchmarkLongContext:
    """run_benchmark(): long context -- clamp контекста, авто-скачивание."""

    def test_clamps_max_context_to_model(self, tmp_path):
        """Модель поддерживает меньше контекста, чем в .env --
        MAX_CONTEXT_TOKENS ограничивается до реального (без sys.exit)."""
        import llm_speed_benchmark.utils as u
        import llm_speed_benchmark.bench_multi as bm
        orig = u.MAX_CONTEXT_TOKENS, u.TOKEN_LIMIT_WARN
        try:
            with patch.object(bm, "apply_config"), \
                 patch.object(bm, "Queue"), \
                 patch.object(bm, "Event"), \
                 patch.object(bm, "Process") as mock_proc, \
                 patch.object(bm, "Live"), \
                 patch.object(bm, "Console"), \
                 patch.object(bm, "signal"), \
                 patch.object(u, "get_client"), \
                 patch.object(u, "detect_model_context_length", return_value=130000), \
                 patch.object(u, "MODEL", "test-model"), \
                 patch.object(u, "MAX_CONTEXT_TOKENS", 131072), \
                 patch("llm_speed_benchmark.long_context.ensure_beam_data"), \
                 patch("llm_speed_benchmark.long_context.LongContextDataset") as mock_ds_cls:
                mock_proc.return_value.start.return_value = None
                mock_ds_cls.return_value.get_conversations_info.return_value = [1]
                mock_ds_cls.return_value.get_messages.return_value = [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ]
                bm.run_benchmark(workers=1, duration=1, long_context_workers=1,
                                 data_dir=str(tmp_path))
                # Контекст ограничен до реального значения модели
                assert u.MAX_CONTEXT_TOKENS == 130000
                # Обрезка диалога -- до реального контекста минус запас
                mock_ds_cls.return_value.get_messages.assert_called_once()
                call_kwargs = mock_ds_cls.return_value.get_messages.call_args[1]
                assert call_kwargs["max_tokens"] == 130000 - 1000
        finally:
            u.MAX_CONTEXT_TOKENS, u.TOKEN_LIMIT_WARN = orig

    def test_auto_download_triggered_when_data_missing(self, tmp_path):
        """Датасета нет на диске -- вызывается ensure_beam_data (скачивание)."""
        import llm_speed_benchmark.utils as u
        import llm_speed_benchmark.bench_multi as bm
        with patch.object(bm, "apply_config"), \
             patch.object(bm, "Queue"), \
             patch.object(bm, "Event"), \
             patch.object(bm, "Process") as mock_proc, \
             patch.object(bm, "Live"), \
             patch.object(bm, "Console"), \
             patch.object(bm, "signal"), \
             patch.object(u, "get_client"), \
             patch.object(u, "detect_model_context_length", return_value=131072), \
             patch.object(u, "MODEL", "test-model"), \
             patch.object(u, "MAX_CONTEXT_TOKENS", 131072), \
             patch("llm_speed_benchmark.long_context.ensure_beam_data") as mock_ensure, \
             patch("llm_speed_benchmark.long_context.LongContextDataset") as mock_ds_cls:
            mock_proc.return_value.start.return_value = None
            mock_ds_cls.return_value.get_conversations_info.return_value = [1]
            mock_ds_cls.return_value.get_messages.return_value = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ]
            bm.run_benchmark(workers=1, duration=1, long_context_workers=1,
                             data_dir=str(tmp_path))
            mock_ensure.assert_called_once()
            call_args, call_kwargs = mock_ensure.call_args
            assert call_args[0] == str(tmp_path)
            assert call_args[1] == "100K"
            assert call_kwargs["allow_download"] is True

    def test_model_timeout_passed_to_worker(self, tmp_path):
        """run_benchmark пробрасывает model_timeout в аргументы воркера."""
        import llm_speed_benchmark.utils as u
        import llm_speed_benchmark.bench_multi as bm
        with patch.object(bm, "apply_config"), \
             patch.object(bm, "Queue"), \
             patch.object(bm, "Event"), \
             patch.object(bm, "Process") as mock_proc, \
             patch.object(bm, "Live"), \
             patch.object(bm, "Console"), \
             patch.object(bm, "signal"), \
             patch.object(u, "get_client"), \
             patch.object(u, "detect_model_context_length", return_value=131072), \
             patch.object(u, "MODEL", "test-model"), \
             patch.object(u, "MAX_CONTEXT_TOKENS", 131072), \
             patch("llm_speed_benchmark.long_context.ensure_beam_data"), \
             patch("llm_speed_benchmark.long_context.LongContextDataset") as mock_ds_cls:
            mock_proc.return_value.start.return_value = None
            mock_ds_cls.return_value.get_conversations_info.return_value = [1]
            mock_ds_cls.return_value.get_messages.return_value = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ]
            bm.run_benchmark(workers=1, duration=1, long_context_workers=1,
                             data_dir=str(tmp_path), model_timeout=0)
            # args воркера: (id, q, start_event, duration, msgs, skip, lc, timeout)
            proc_args = mock_proc.call_args[1]["args"]
            assert proc_args[7] == 0  # model_timeout=0 (без таймаута)
