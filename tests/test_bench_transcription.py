"""
tests/test_bench_transcription.py

Тесты для bench_transcription и transcription_utils.
Не требуют запущенного сервера.
"""

import io
import json
import wave
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, patch


class TestTranscriptionUtils:
    """Тесты для transcription_utils."""

    def test_get_audio_duration_wav(self, tmp_path):
        from llm_speed_benchmark.transcription_utils import get_audio_duration

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframesraw(b"\x00" * 32000)  # 1 sec
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(buf.getvalue())

        dur = get_audio_duration(str(wav_file))
        assert abs(dur - 1.0) < 0.01

    def test_get_audio_duration_half_sec(self, tmp_path):
        from llm_speed_benchmark.transcription_utils import get_audio_duration

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframesraw(b"\x00" * 16000)  # 0.5 sec
        wav_file = tmp_path / "half.wav"
        wav_file.write_bytes(buf.getvalue())

        dur = get_audio_duration(str(wav_file))
        assert abs(dur - 0.5) < 0.01

    def test_get_audio_info(self, tmp_path):
        from llm_speed_benchmark.transcription_utils import get_audio_info

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframesraw(b"\x00" * 176400)  # 1 sec stereo
        wav_file = tmp_path / "info.wav"
        wav_file.write_bytes(buf.getvalue())

        info = get_audio_info(str(wav_file))
        assert abs(info["duration"] - 1.0) < 0.01
        assert info["channels"] == 2.0
        assert info["sample_rate"] == 44100.0
        assert info["sampwidth"] == 2.0
        assert info["size_bytes"] > 0

    def test_load_bundled_audio(self):
        from llm_speed_benchmark.transcription_utils import load_bundled_audio

        paths = load_bundled_audio()
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()
            assert p.suffix == ".wav"

    def test_generate_audio_at_duration(self):
        from llm_speed_benchmark.transcription_utils import (
            generate_audio_at_duration,
            get_audio_duration,
            load_bundled_audio,
        )

        bundled = load_bundled_audio()
        target = 2.0
        path = generate_audio_at_duration(target, bundled_audios=bundled)

        assert path.exists()
        actual = get_audio_duration(path)
        assert abs(actual - target) < 0.1

    def test_generate_audio_at_duration_caching(self):
        from llm_speed_benchmark.transcription_utils import (
            generate_audio_at_duration,
            load_bundled_audio,
        )

        bundled = load_bundled_audio()
        path1 = generate_audio_at_duration(1.0, bundled_audios=bundled)
        path2 = generate_audio_at_duration(1.0, bundled_audios=bundled)
        assert path1 == path2  # cached

    def test_generate_duration_series(self):
        from llm_speed_benchmark.transcription_utils import (
            generate_duration_series,
            load_bundled_audio,
        )

        bundled = load_bundled_audio()
        durations = [0.5, 1.0, 2.0]
        series = generate_duration_series(durations, bundled_audios=bundled)

        assert len(series) == 3
        for path, actual_dur in series:
            assert path.exists()

    def test_discover_audio_files(self, tmp_path):
        from llm_speed_benchmark.transcription_utils import discover_audio_files

        for name in ["audio1.wav", "audio2.mp3", "readme.txt"]:
            (tmp_path / name).write_text("dummy")

        found = discover_audio_files(str(tmp_path))
        stems = {p.stem for p in found}
        assert "audio1" in stems
        assert "audio2" in stems
        assert "readme" not in stems

    def test_discover_audio_files_empty_dir(self, tmp_path):
        from llm_speed_benchmark.transcription_utils import discover_audio_files

        found = discover_audio_files(str(tmp_path))
        assert found == []

    def test_discover_audio_files_nonexistent(self):
        from llm_speed_benchmark.transcription_utils import discover_audio_files

        found = discover_audio_files("/nonexistent/path")
        assert found == []

    def test_filter_by_max_duration(self, tmp_path):
        from llm_speed_benchmark.transcription_utils import filter_by_max_duration

        for secs in [0.5, 2.0, 5.0]:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframesraw(b"\x00" * int(16000 * 2 * secs))
            (tmp_path / f"audio_{secs}s.wav").write_bytes(buf.getvalue())

        all_paths = sorted(tmp_path.glob("*.wav"))
        filtered = filter_by_max_duration(all_paths, max_duration=3.0)
        assert len(filtered) == 2  # 0.5s and 2.0s

    def test_filter_by_max_duration_no_limit(self, tmp_path):
        from llm_speed_benchmark.transcription_utils import filter_by_max_duration

        for i in range(3):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframesraw(b"\x00" * 32000)
            (tmp_path / f"audio_{i}.wav").write_bytes(buf.getvalue())

        all_paths = sorted(tmp_path.glob("*.wav"))
        filtered = filter_by_max_duration(all_paths, max_duration=None)
        assert len(filtered) == 3

    def test_build_transcription_request(self):
        from llm_speed_benchmark.transcription_utils import build_transcription_request

        path, params = build_transcription_request(
            "/tmp/test.wav",
            model="whisper-small",
            prompt="Context here",
            language="ru",
        )

        assert path == Path("/tmp/test.wav")
        assert params["model"] == "whisper-small"
        assert params["prompt"] == "Context here"
        assert params["language"] == "ru"
        assert params["response_format"] == "json"

    def test_build_transcription_request_minimal(self):
        from llm_speed_benchmark.transcription_utils import build_transcription_request

        path, params = build_transcription_request(
            "/tmp/test.wav",
            model="whisper",
        )

        assert path == Path("/tmp/test.wav")
        assert params == {
            "model": "whisper",
            "response_format": "json",
        }


class TestTranscriptionWorker:
    """Тесты для воркера bench_transcription (моки OpenAI)."""

    def _make_wav(self, tmp_path, duration: float) -> Path:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframesraw(b"\x00" * int(16000 * 2 * duration))
        wav = tmp_path / f"audio_{duration}s.wav"
        wav.write_bytes(buf.getvalue())
        return wav

    def _run_worker(
        self, tmp_path, duration_sec, model_timeout, skip_errors, mode,
        mock_create_func, duration_arg=None,
    ):
        """Запускает _worker в потоке с start_event."""
        from multiprocessing import Event, Queue

        from llm_speed_benchmark.bench_transcription import _worker

        wav = self._make_wav(tmp_path, duration_sec)
        audio_tasks = [{"path": str(wav), "duration": duration_sec}]

        q = Queue()
        start_event = Event()

        with patch("llm_speed_benchmark.utils.BASE_URL", "http://test"), \
             patch("llm_speed_benchmark.utils.API_KEY", "test"), \
             patch("llm_speed_benchmark.utils.MODEL", "whisper"), \
             patch("llm_speed_benchmark.bench_transcription.OpenAI") as MockOpenAI:

            mock_client = MagicMock()
            mock_client.audio.transcriptions.create = mock_create_func
            MockOpenAI.return_value = mock_client

            def run_it():
                start_event.set()
                _worker(
                    worker_id=0, q=q, start_event=start_event,
                    audio_tasks=audio_tasks, duration=duration_arg,
                    model_timeout=model_timeout, prompt=None, language=None,
                    response_format="json", skip_errors=skip_errors,
                    response_width=60, mode=mode,
                )

            t = Thread(target=run_it, daemon=True)
            t.start()
            t.join(timeout=15)

        messages = []
        while not q.empty():
            try:
                messages.append(q.get_nowait())
            except Exception:
                break
        return messages

    def test_worker_successful_transcription(self, tmp_path):
        mock_response = MagicMock()
        mock_response.text = "Privet mir"

        messages = self._run_worker(
            tmp_path, 1.0, 60, skip_errors=True, mode="once",
            mock_create_func=MagicMock(return_value=mock_response),
        )

        start_msgs = [m for m in messages if m.get("type") == "start"]
        assert len(start_msgs) == 1

        stats_msgs = [m for m in messages if m.get("type") == "stats"]
        assert len(stats_msgs) == 1
        stats = stats_msgs[0]
        assert stats["calls"] == 1
        assert stats["audio_dur"] == 1.0
        assert stats["rtf"] >= 0

    def test_worker_error_handling(self, tmp_path):
        messages = self._run_worker(
            tmp_path, 1.0, 60, skip_errors=False, mode="once",
            mock_create_func=MagicMock(side_effect=Exception("Connection error")),
        )

        error_msgs = [m for m in messages if m.get("type") == "error_stop"]
        assert len(error_msgs) == 1
        assert "Connection error" in error_msgs[0]["error"]

    def test_worker_skip_errors(self, tmp_path):
        messages = self._run_worker(
            tmp_path, 1.0, 60, skip_errors=True, mode="once",
            mock_create_func=MagicMock(side_effect=Exception("API down")),
        )

        error_stop_msgs = [m for m in messages if m.get("type") == "error_stop"]
        assert len(error_stop_msgs) == 0

        # With skip_errors=True, worker sends stats with error content
        stats_msgs = [m for m in messages if m.get("type") == "stats"]
        assert len(stats_msgs) >= 1
        # The stats should have 0 tokens (error case)
        assert stats_msgs[-1]["call_tokens"] == 0

    def test_worker_cycling_mode(self, tmp_path):
        call_counter = [0]
        def mock_create(**kwargs):
            call_counter[0] += 1
            mock_response = MagicMock()
            mock_response.text = f"Transcription {call_counter[0]}"
            return mock_response

        messages = self._run_worker(
            tmp_path, 0.5, 60, skip_errors=True, mode="cycling",
            mock_create_func=mock_create,
            duration_arg=2,
        )

        stats_msgs = [m for m in messages if m.get("type") == "stats"]
        assert len(stats_msgs) >= 1
        if len(stats_msgs) >= 1:
            assert stats_msgs[-1]["calls"] == len(stats_msgs)


class TestTranscriptionLiveTable:
    """Тесты для TranscriptionLiveTable."""

    def test_table_init(self):
        from llm_speed_benchmark.bench_transcription import TranscriptionLiveTable

        table = TranscriptionLiveTable(duration=60, total_workers=4)
        assert table.total_workers == 4
        assert table.duration == 60
        assert table.workers == {}

    def test_table_mark_started(self):
        from llm_speed_benchmark.bench_transcription import TranscriptionLiveTable

        table = TranscriptionLiveTable(duration=None, total_workers=2)
        table.mark_started(0, task_count=10)
        table.mark_started(1, task_count=10)

        assert 0 in table.workers
        assert 1 in table.workers
        assert table.workers[0]["total_tasks"] == 10

    def test_table_update_stats(self):
        from llm_speed_benchmark.bench_transcription import TranscriptionLiveTable

        table = TranscriptionLiveTable(duration=None, total_workers=1)
        table.mark_started(0)

        table.update_stats({
            "type": "stats",
            "id": 0,
            "calls": 5,
            "tokens": 150,
            "audio_dur": 10.0,
            "proc_time": 3.5,
            "rtf": 0.35,
            "avg_rtf": 0.35,
            "tok_speed": 15.0,
            "audio_speed": 2.86,
            "wall": "00:30",
            "tail": "Privet mir",
            "file": "test_audio",
        })

        w = table.workers[0]
        assert w["calls"] == 5
        assert w["tokens"] == 150
        assert w["rtf"] == 0.35
        assert w["wall"] == "00:30"

    def test_table_render(self):
        from llm_speed_benchmark.bench_transcription import TranscriptionLiveTable

        table = TranscriptionLiveTable(duration=None, total_workers=1)
        table.mark_started(0)

        rendered = table.render()
        assert rendered is not None

    def test_table_render_with_error(self):
        from llm_speed_benchmark.bench_transcription import TranscriptionLiveTable

        table = TranscriptionLiveTable(duration=None, total_workers=2)
        table.mark_started(0)
        table.mark_started(1)
        table.mark_error(1, "Traceback...")

        rendered = table.render()
        assert rendered is not None

    def test_table_render_with_stopped(self):
        from llm_speed_benchmark.bench_transcription import TranscriptionLiveTable

        table = TranscriptionLiveTable(duration=None, total_workers=1)
        table.mark_started(0)
        table.mark_stopped(0, "Connection refused")

        rendered = table.render()
        assert rendered is not None


class TestTranscriptionCLI:
    """Тесты для CLI bench_transcription."""

    def test_cli_help(self):
        from llm_speed_benchmark.bench_transcription import cli

        with patch("sys.argv", ["bench_transcription", "--help"]):
            try:
                cli()
            except SystemExit:
                pass

    def _make_test_wav(self, tmp_path):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframesraw(b"\x00" * 32000)
        wav = tmp_path / "test.wav"
        wav.write_bytes(buf.getvalue())
        return wav

    def test_cli_parsing(self, tmp_path):
        from llm_speed_benchmark.bench_transcription import cli

        self._make_test_wav(tmp_path)

        with patch("sys.argv", [
            "bench_transcription",
            "--audio", str(tmp_path),
            "--model", "whisper-small",
            "--base-url", "http://localhost:8000/v1",
            "--workers", "2",
            "--duration", "30",
            "--language", "ru",
            "--mode", "once",
        ]):
            with patch("llm_speed_benchmark.bench_transcription.run_benchmark") as mock_run:
                cli()
                assert mock_run.called
                kwargs = mock_run.call_args.kwargs
                assert kwargs["model"] == "whisper-small"
                assert kwargs["workers"] == 2
                assert kwargs["duration"] == 30
                assert kwargs["language"] == "ru"
                assert kwargs["mode"] == "once"

    def test_cli_durations(self, tmp_path):
        from llm_speed_benchmark.bench_transcription import cli

        with patch("sys.argv", [
            "bench_transcription",
            "--durations", "5", "10", "30", "60",
        ]):
            with patch("llm_speed_benchmark.bench_transcription.run_benchmark") as mock_run:
                cli()
                kwargs = mock_run.call_args.kwargs
                assert kwargs["durations"] == [5.0, 10.0, 30.0, 60.0]

    def test_cli_max_duration(self):
        from llm_speed_benchmark.bench_transcription import cli

        with patch("sys.argv", [
            "bench_transcription",
            "--max-duration", "30",
        ]):
            with patch("llm_speed_benchmark.bench_transcription.run_benchmark") as mock_run:
                cli()
                kwargs = mock_run.call_args.kwargs
                assert kwargs["max_duration"] == 30.0

    def test_cli_model_timeout_env(self, tmp_path, monkeypatch):
        from llm_speed_benchmark.bench_transcription import cli

        monkeypatch.setenv("TRANSCRIPTION_MODEL_TIMEOUT", "1200")
        self._make_test_wav(tmp_path)

        with patch("sys.argv", [
            "bench_transcription",
            "--audio", str(tmp_path),
            "--mode", "once",
        ]):
            with patch("llm_speed_benchmark.bench_transcription.run_benchmark") as mock_run:
                cli()
                kwargs = mock_run.call_args.kwargs
                assert kwargs["model_timeout"] == 1200.0

    def test_cli_model_timeout_override(self, tmp_path, monkeypatch):
        from llm_speed_benchmark.bench_transcription import cli

        monkeypatch.setenv("TRANSCRIPTION_MODEL_TIMEOUT", "1200")
        self._make_test_wav(tmp_path)

        with patch("sys.argv", [
            "bench_transcription",
            "--audio", str(tmp_path),
            "--model-timeout", "300",
            "--mode", "once",
        ]):
            with patch("llm_speed_benchmark.bench_transcription.run_benchmark") as mock_run:
                cli()
                kwargs = mock_run.call_args.kwargs
                assert kwargs["model_timeout"] == 300.0


class TestEnvPriority:
    """Тесты приоритета TRANSCRIPTION_* над BASE_URL/MODEL."""

    def test_transcription_vars_override_base(self, tmp_path, monkeypatch):
        """TRANSCRIPTION_BASE_URL перезаписывает BASE_URL для bench_transcription."""
        import llm_speed_benchmark.utils as u

        orig_base = u.BASE_URL
        orig_key = u.API_KEY
        orig_model = u.MODEL

        try:
            # Симулируем: .env дал BASE_URL, TRANSCRIPTION_* свои
            monkeypatch.setenv("BASE_URL", "http://base-url/v1")
            monkeypatch.setenv("API_KEY", "base-key")
            monkeypatch.setenv("MODEL", "base-model")
            monkeypatch.setenv("TRANSCRIPTION_BASE_URL", "http://asr-url/v1")
            monkeypatch.setenv("TRANSCRIPTION_API_KEY", "asr-key")
            monkeypatch.setenv("TRANSCRIPTION_MODEL", "asr-model")

            # run_benchmark читает TRANSCRIPTION_* если CLI не указал
            from llm_speed_benchmark.bench_transcription import run_benchmark

            captured = {}
            def mock_apply_config(base_url=None, api_key=None, model=None, max_context=None):
                captured["base_url"] = base_url
                captured["api_key"] = api_key
                captured["model"] = model

            with patch("llm_speed_benchmark.bench_transcription.apply_config", mock_apply_config), \
                 patch("llm_speed_benchmark.bench_transcription.load_bundled_audio", return_value=[]):

                try:
                    run_benchmark(workers=1, mode="once")
                except SystemExit:
                    pass  # no audio files, expected

            assert captured["base_url"] == "http://asr-url/v1"
            assert captured["api_key"] == "asr-key"
            assert captured["model"] == "asr-model"
        finally:
            u.BASE_URL = orig_base
            u.API_KEY = orig_key
            u.MODEL = orig_model

    def test_cli_overrides_transcription_vars(self, tmp_path, monkeypatch):
        """CLI аргументы имеют приоритет над TRANSCRIPTION_*."""
        import llm_speed_benchmark.utils as u

        orig_base = u.BASE_URL
        orig_key = u.API_KEY
        orig_model = u.MODEL

        try:
            monkeypatch.setenv("TRANSCRIPTION_BASE_URL", "http://asr-url/v1")
            monkeypatch.setenv("TRANSCRIPTION_API_KEY", "asr-key")
            monkeypatch.setenv("TRANSCRIPTION_MODEL", "asr-model")

            captured = {}
            def mock_apply_config(base_url=None, api_key=None, model=None, max_context=None):
                captured["base_url"] = base_url
                captured["api_key"] = api_key
                captured["model"] = model

            from llm_speed_benchmark.bench_transcription import run_benchmark

            with patch("llm_speed_benchmark.bench_transcription.apply_config", mock_apply_config), \
                 patch("llm_speed_benchmark.bench_transcription.load_bundled_audio", return_value=[]):

                try:
                    run_benchmark(
                        workers=1, mode="once",
                        base_url="http://cli-url/v1",
                        api_key="cli-key",
                        model="cli-model",
                    )
                except SystemExit:
                    pass

            assert captured["base_url"] == "http://cli-url/v1"
            assert captured["api_key"] == "cli-key"
            assert captured["model"] == "cli-model"
        finally:
            u.BASE_URL = orig_base
            u.API_KEY = orig_key
            u.MODEL = orig_model

    def test_worker_uses_transcription_model(self, tmp_path, monkeypatch):
        """Интеграционный: воркер получает TRANSCRIPTION_MODEL, а не MODEL из .env."""
        import os
        from multiprocessing import Event, Queue

        import llm_speed_benchmark.utils as u
        from llm_speed_benchmark.bench_transcription import _worker

        orig_u_model = u.MODEL

        try:
            # Создаём тестовый WAV
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframesraw(b"\x00" * 32000)
            wav = tmp_path / "test.wav"
            wav.write_bytes(buf.getvalue())

            audio_tasks = [{"path": str(wav), "duration": 1.0}]
            q = Queue()
            start_event = Event()

            received_model = [None]

            def mock_create(**kwargs):
                received_model[0] = kwargs.get("model")
                mock_response = MagicMock()
                mock_response.text = "Transcribed text"
                return mock_response

            # Патчим utils.MODEL на TRANSCRIPTION_MODEL
            # Важно: патчим OpenAI в bench_transcription (не в openai!)
            # т.к. воркер импортирует `from openai import OpenAI` на уровне модуля
            with patch.object(u, "BASE_URL", "http://test/v1"), \
                 patch.object(u, "API_KEY", "test"), \
                 patch.object(u, "MODEL", "Qwen3-ASR-0.6B"), \
                 patch("llm_speed_benchmark.bench_transcription.OpenAI") as MockOpenAI:

                mock_client = MagicMock()
                mock_client.audio.transcriptions.create = mock_create
                MockOpenAI.return_value = mock_client

                def run_it():
                    start_event.set()
                    _worker(
                        worker_id=0, q=q, start_event=start_event,
                        audio_tasks=audio_tasks, duration=None,
                        model_timeout=60, prompt=None, language=None,
                        response_format="json", skip_errors=True,
                        response_width=60, mode="once",
                    )

                t = Thread(target=run_it, daemon=True)
                t.start()
                t.join(timeout=10)

            assert received_model[0] == "Qwen3-ASR-0.6B", \
                f"Worker used model '{received_model[0]}' instead of 'Qwen3-ASR-0.6B'"

        finally:
            u.MODEL = orig_u_model