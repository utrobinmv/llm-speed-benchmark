"""
tests/test_bench_audio.py

Тесты для bench_audio и audio_utils.
Не требуют запущенного сервера.
"""

import base64
from unittest.mock import MagicMock, patch


class TestAudioUtils:
    """Тесты для audio_utils."""

    def test_get_bundled_audio_paths(self):
        from llm_speed_benchmark.audio_utils import get_bundled_audio_paths

        paths = get_bundled_audio_paths()
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()
            assert p.suffix == ".wav"
            assert p.stat().st_size > 0

    def test_get_bundled_audio_paths_sorted(self):
        from llm_speed_benchmark.audio_utils import get_bundled_audio_paths

        paths = get_bundled_audio_paths()
        assert paths == sorted(paths)

    def test_get_audio_paths_default(self):
        from llm_speed_benchmark.audio_utils import get_audio_paths

        paths = get_audio_paths()
        assert len(paths) >= 1

    def test_get_audio_paths_count_limit(self):
        from llm_speed_benchmark.audio_utils import get_audio_paths

        paths = get_audio_paths(count=3)
        assert len(paths) == 3

    def test_get_audio_paths_custom_dir(self, tmp_path):
        from llm_speed_benchmark.audio_utils import get_audio_paths

        # Создаём тестовые WAV файлы
        import wave
        import io

        for i in range(3):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframesraw(b"\x00" * 3200)
            (tmp_path / f"audio_{i}.wav").write_bytes(buf.getvalue())

        paths = get_audio_paths(audio_dir=str(tmp_path))
        assert len(paths) == 3

    def test_load_audio_as_base64(self):
        from llm_speed_benchmark.audio_utils import (
            get_bundled_audio_paths,
            load_audio_as_base64,
        )

        bundled = get_bundled_audio_paths()
        fmt, b64 = load_audio_as_base64(bundled[0])
        assert fmt == "wav"
        assert isinstance(b64, str)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_build_audio_message(self):
        from llm_speed_benchmark.audio_utils import (
            build_audio_message,
            get_bundled_audio_paths,
        )

        bundled = get_bundled_audio_paths()
        messages = build_audio_message([bundled[0]], "Транскрибируй")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Транскрибируй"
        assert content[1]["type"] == "audio_url"
        assert "url" in content[1]["audio_url"]

    def test_build_audio_message_multiple(self):
        from llm_speed_benchmark.audio_utils import (
            build_audio_message,
            get_bundled_audio_paths,
        )

        bundled = get_bundled_audio_paths()
        messages = build_audio_message(
            [bundled[0], bundled[1], bundled[2]], "Транскрибируй"
        )

        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 4  # 1 text + 3 audio
        assert content[0]["type"] == "text"
        for i in range(1, 4):
            assert content[i]["type"] == "audio_url"

    def test_discover_audio(self, tmp_path):
        from llm_speed_benchmark.audio_utils import discover_audio

        import wave
        import io

        for i in range(5):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframesraw(b"\x00" * 3200)
            (tmp_path / f"audio_{i}.wav").write_bytes(buf.getvalue())

        found = discover_audio(tmp_path)
        assert len(found) == 5

    def test_discover_audio_empty(self, tmp_path):
        from llm_speed_benchmark.audio_utils import discover_audio

        found = discover_audio(tmp_path)
        assert found == []

    def test_discover_audio_nonexistent(self):
        from llm_speed_benchmark.audio_utils import discover_audio

        found = discover_audio("/nonexistent/path")
        assert found == []

    def test_discover_audio_ignores_other_files(self, tmp_path):
        from llm_speed_benchmark.audio_utils import discover_audio

        (tmp_path / "image.png").touch()
        (tmp_path / "text.txt").touch()
        (tmp_path / "script.py").touch()

        found = discover_audio(tmp_path)
        assert found == []

    def test_get_audio_info_wav(self):
        from llm_speed_benchmark.audio_utils import (
            get_audio_info,
            get_bundled_audio_paths,
        )

        bundled = get_bundled_audio_paths()
        info = get_audio_info(bundled[0])
        assert "channels" in info
        assert "sample_rate" in info
        assert "frames" in info
        assert "duration" in info
        assert info["duration"] > 0


class TestAudioWorker:
    """Тесты для _worker с моками."""

    @patch("llm_speed_benchmark.bench_audio.OpenAI")
    @patch("llm_speed_benchmark.bench_audio.StreamSession")
    def test_worker_sends_start_message(
        self, mock_session_cls, mock_openai_cls
    ):
        from multiprocessing import Event, Queue

        from llm_speed_benchmark.audio_utils import get_bundled_audio_paths
        from llm_speed_benchmark.bench_audio import _worker

        q = Queue()
        start_event = Event()
        start_event.set()

        bundled = get_bundled_audio_paths()
        audio_paths = [str(p) for p in bundled[:2]]

        mock_session = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.completion_tokens = 100
        mock_metrics.prompt_tokens = 50
        mock_metrics.chunk_count = 50
        mock_metrics.elapsed = 1.0
        mock_metrics.ttft = 0.5
        mock_metrics.assistant_content = "Это транскрипция аудио."
        mock_metrics.call_speed = 100.0
        mock_metrics.instant_speed = 100.0
        mock_session.run.return_value = mock_metrics
        mock_session.tokens_per_chunk = 2.0
        mock_session_cls.return_value = mock_session

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        p = __import__("threading", fromlist=["Thread"]).Thread(
            target=_worker,
            args=(0, q, start_event, audio_paths, 2, ["Транскрибируй"], 1, False, 60),
            daemon=True,
        )
        p.start()
        p.join(timeout=10)

        messages = []
        while not q.empty():
            try:
                messages.append(q.get_nowait())
            except Exception:
                break

        start_msgs = [m for m in messages if m.get("type") == "start"]
        assert len(start_msgs) == 1
        assert start_msgs[0]["id"] == 0

    @patch("llm_speed_benchmark.bench_audio.OpenAI")
    @patch("llm_speed_benchmark.bench_audio.StreamSession")
    def test_worker_sends_stats(
        self, mock_session_cls, mock_openai_cls
    ):
        from multiprocessing import Event, Queue

        from llm_speed_benchmark.audio_utils import get_bundled_audio_paths
        from llm_speed_benchmark.bench_audio import _worker

        q = Queue()
        start_event = Event()
        start_event.set()

        bundled = get_bundled_audio_paths()
        audio_paths = [str(bundled[0])]

        mock_session = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.completion_tokens = 80
        mock_metrics.prompt_tokens = 30
        mock_metrics.chunk_count = 40
        mock_metrics.elapsed = 0.8
        mock_metrics.ttft = 0.3
        mock_metrics.assistant_content = "Транскрипция аудио файла."
        mock_metrics.call_speed = 100.0
        mock_metrics.instant_speed = 100.0
        mock_session.run.return_value = mock_metrics
        mock_session.tokens_per_chunk = 2.0
        mock_session_cls.return_value = mock_session

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        p = __import__("threading", fromlist=["Thread"]).Thread(
            target=_worker,
            args=(1, q, start_event, audio_paths, 2, ["Транскрибируй"], 1, False, 60),
            daemon=True,
        )
        p.start()
        p.join(timeout=10)

        messages = []
        while not q.empty():
            try:
                messages.append(q.get_nowait())
            except Exception:
                break

        stats_msgs = [m for m in messages if m.get("type") == "stats"]
        assert len(stats_msgs) >= 1
        assert stats_msgs[0]["id"] == 1
        assert stats_msgs[0]["cg"] == 80


class TestAudioLiveTable:
    """Тесты для AudioLiveTable."""

    def test_render_empty(self):
        from llm_speed_benchmark.bench_audio import AudioLiveTable

        table = AudioLiveTable(None, 4)
        rendered = table.render()
        assert rendered is not None

    def test_render_with_workers(self):
        from llm_speed_benchmark.bench_audio import AudioLiveTable

        table = AudioLiveTable(None, 2, response_width=40)
        table.mark_started(0, 10)
        table.mark_started(1, 10)

        table.update_stats({
            "id": 0,
            "calls": 5,
            "media": "test_audio_000",
            "media_count": 1,
            "g": 500,
            "cg": 100,
            "speed": 50.0,
            "avg_speed": 45.0,
            "inst_speed": 55.0,
            "ttft": 0.5,
            "ttft_sum": 2.5,
            "tail": "Транскрипция аудио файла с голосом",
            "wall": "00:10",
        })

        rendered = table.render()
        assert rendered is not None

    def test_update_stats_maps_g_to_gen(self):
        """REGRESSION: 'g' в update_stats маппится на 'gen'."""
        from llm_speed_benchmark.bench_audio import AudioLiveTable

        table = AudioLiveTable(None, 1)
        table.mark_started(0, 10)
        table.update_stats({
            "id": 0,
            "calls": 3,
            "g": 1500,
            "cg": 500,
            "avg_speed": 30.0,
            "ttft": 0.5,
            "ttft_sum": 1.5,
            "wall": "00:10",
            "tail": "response text",
        })
        w = table.workers[0]
        assert w["gen"] == 1500, f"gen должен быть 1500, получен {w.get('gen', 'MISSING')}"
        assert w["calls"] == 3

    def test_clean_tail_empty(self):
        from llm_speed_benchmark.bench_audio import AudioLiveTable

        table = AudioLiveTable(duration=None, total_workers=1, response_width=20)
        assert table._clean_tail("") == ""

    def test_clean_tail_truncation(self):
        from llm_speed_benchmark.bench_audio import AudioLiveTable

        table = AudioLiveTable(duration=None, total_workers=1, response_width=20)
        long_text = "A" * 100
        result = table._clean_tail(long_text)
        assert len(result) <= 25  # ... + max_vlen + небольшой запас
        assert result.startswith("...")


class TestAudioCli:
    """bench_audio CLI: парсинг аргументов."""

    def test_cli_default_args(self):
        from llm_speed_benchmark.bench_audio import cli

        with patch("sys.argv", ["bench_audio"]):
            with patch("llm_speed_benchmark.bench_audio.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["workers"] == 4
                assert call_kwargs["max_audio"] == 1

    def test_cli_custom_workers(self):
        from llm_speed_benchmark.bench_audio import cli

        with patch("sys.argv", ["bench_audio", "-w", "2", "-d", "30"]):
            with patch("llm_speed_benchmark.bench_audio.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["workers"] == 2
                assert call_kwargs["duration"] == 30

    def test_cli_skip_errors(self):
        from llm_speed_benchmark.bench_audio import cli

        with patch("sys.argv", ["bench_audio", "--skip-errors"]):
            with patch("llm_speed_benchmark.bench_audio.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["skip_errors"] is True

    def test_cli_custom_prompts(self):
        from llm_speed_benchmark.bench_audio import cli

        with patch("sys.argv", ["bench_audio", "-p", "Prompt 1", "-p", "Prompt 2"]):
            with patch("llm_speed_benchmark.bench_audio.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["prompts"] == ["Prompt 1", "Prompt 2"]

    def test_cli_max_audio(self):
        from llm_speed_benchmark.bench_audio import cli

        with patch("sys.argv", ["bench_audio", "--max-audio", "3"]):
            with patch("llm_speed_benchmark.bench_audio.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["max_audio"] == 3