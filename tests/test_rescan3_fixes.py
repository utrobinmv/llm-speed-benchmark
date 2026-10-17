#!/usr/bin/env python3
"""
tests/test_rescan3_fixes.py

Регрессионные тесты по третьему прогону ocr scan (rescan3):
- live_table._clean_tail: литеральные '[' в выводе LLM не должны
  открывать тег и «проглатывать» остаток строки.
- audio_utils.get_audio_paths: explicit count=0 даёт пустой список,
  а не «все файлы».
- streaming: fallback completion_tokens==0 -> chunk_count срабатывает
  только когда usage вообще не пришло от API.
"""

from unittest.mock import MagicMock
import wave

# tmp_path — фикстура pytest, поставляется автоматически;
# pytest импортируется неявно (используется pyt-раннером).


# ---------------------------------------------------------------------------
# live_table._clean_tail -- литеральные скобки
# ---------------------------------------------------------------------------

class TestCleanTailLiteralBrackets:
    """Литеральные '[' в LLM-выводе (JSON/code) не должны ломать разметку."""

    def _make_table(self):
        # Реализация live-table живёт в bench_multi (LiveTable) и
        # наследует _clean_tail из BaseLiveTable.
        from llm_speed_benchmark.bench_multi import LiveTable
        return LiveTable(duration=None, total_workers=1, response_width=120)

    def test_json_brackets_preserved(self):
        t = self._make_table()
        out = t._clean_tail('Ответ: [1, 2, 3] конец', 60)
        assert "[1, 2, 3]" in out
        assert "Ответ" in out
        assert "конец" in out

    def test_literal_bracket_does_not_swallow_tail(self):
        t = self._make_table()
        # незакрытая '[' не должна проглотить остаток строки
        out = t._clean_tail('код [abc', 60)
        assert "код" in out
        assert "abc" in out

    def test_real_tags_kept(self):
        # настоящий Rich-тег [red]...[/] сохраняется как тег
        t = self._make_table()
        out = t._clean_tail('[red]ошибка[/] тут', 60)
        assert "ошибка" in out
        assert "тут" in out


# ---------------------------------------------------------------------------
# audio_utils.get_audio_paths -- count=0
# ---------------------------------------------------------------------------

def _make_wavs(tmp_path, names):
    for name in names:
        p = tmp_path / name
        with wave.open(str(p), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 100)


class TestAudioPathsCount:
    """count=0 должен давать пустой список, а не «все файлы»."""

    def test_count_zero_returns_empty(self, tmp_path):
        from llm_speed_benchmark.audio_utils import get_audio_paths
        _make_wavs(tmp_path, ("a.wav", "b.wav"))
        paths = get_audio_paths(audio_dir=str(tmp_path), count=0)
        assert paths == []

    def test_count_none_returns_all(self, tmp_path):
        from llm_speed_benchmark.audio_utils import get_audio_paths
        _make_wavs(tmp_path, ("a.wav", "b.wav"))
        paths = get_audio_paths(audio_dir=str(tmp_path), count=None)
        assert len(paths) == 2

    def test_count_positive_limits(self, tmp_path):
        from llm_speed_benchmark.audio_utils import get_audio_paths
        _make_wavs(tmp_path, ("a.wav", "b.wav", "c.wav"))
        paths = get_audio_paths(audio_dir=str(tmp_path), count=2)
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# streaming -- usage-seen fallback
# ---------------------------------------------------------------------------

def _make_chunk(content=None, usage=None, reasoning=None):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.reasoning = reasoning
    delta.reasoning_content = None
    chunk.choices = [MagicMock(delta=delta)]
    chunk.usage = usage
    return chunk


class TestStreamingUsageFallback:
    """completion_tokens==0 -> chunk_count только при отсутствии usage."""

    def _make_client(self, chunks):
        from llm_speed_benchmark.streaming import StreamSession  # noqa
        client = MagicMock()
        client.chat.completions.create.side_effect = lambda **kw: iter(chunks)
        return client

    def test_usage_present_zero_preserved(self):
        """usage пришло и честно дало 0 -> completion остаётся 0
        (пустая генерация не затирается числом чанков)."""
        from unittest.mock import MagicMock
        from llm_speed_benchmark.streaming import StreamSession

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 0

        chunks = [
            _make_chunk(content="", usage=usage),
            _make_chunk(content="", usage=usage),
        ]
        client = self._make_client(chunks)
        session = StreamSession(client)
        m = session.run(
            messages=[{"role": "user", "content": "x"}],
            model="m",
        )
        # usage было (0) -> completion остаётся 0, НЕ равен chunk_count(0)
        assert m.completion_tokens == 0
        assert m.chunk_count == 0

    def test_usage_missing_fallback_chunk_count(self):
        """usage не пришло вовсе -> completion подставляется числом чанков."""
        from llm_speed_benchmark.streaming import StreamSession

        chunks = [
            _make_chunk(content="a", usage=None),
            _make_chunk(content="b", usage=None),
        ]
        client = self._make_client(chunks)
        session = StreamSession(client)
        m = session.run(
            messages=[{"role": "user", "content": "x"}],
            model="m",
        )
        assert m.chunk_count == 2
        # usage не пришло -> fallback: completion = chunk_count = 2
        assert m.completion_tokens == 2

    def test_usage_present_nonzero_kept(self):
        """usage пришло с реальными токенами -> берём их."""
        from llm_speed_benchmark.streaming import StreamSession

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 7

        chunks = [_make_chunk(content="", usage=usage)]
        client = self._make_client(chunks)
        session = StreamSession(client)
        m = session.run(
            messages=[{"role": "user", "content": "x"}],
            model="m",
        )
        assert m.completion_tokens == 7