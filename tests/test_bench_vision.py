"""
tests/test_bench_vision.py

Тесты для bench_vision и image_utils.
Не требуют запущенного сервера.
"""

import base64
from unittest.mock import MagicMock, patch



class TestImageUtils:
    """Тесты для image_utils."""

    def test_generate_test_images(self, tmp_path):
        from llm_speed_benchmark.image_utils import generate_test_images

        images = generate_test_images(str(tmp_path), count=8)
        assert len(images) == 8
        for img in images:
            assert img.exists()
            assert img.suffix == ".png"
            assert img.stat().st_size > 0

    def test_generate_test_images_variety(self, tmp_path):
        from llm_speed_benchmark.image_utils import generate_test_images

        # 8 вариантов * 3 = 24 изображения
        images = generate_test_images(str(tmp_path), count=24)
        assert len(images) == 24

    def test_generate_min_count(self, tmp_path):
        from llm_speed_benchmark.image_utils import generate_test_images

        images = generate_test_images(str(tmp_path), count=2)
        assert len(images) == 4  # минимум 4

    def test_load_image_as_base64(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            generate_test_images,
            load_image_as_base64,
        )

        images = generate_test_images(str(tmp_path), count=1)
        b64 = load_image_as_base64(images[0])
        assert isinstance(b64, str)
        # Проверяем что декодируется
        decoded = base64.b64decode(b64)
        assert decoded.startswith(b"\x89PNG")  # PNG magic

    def test_build_vision_message(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            build_vision_message,
            generate_test_images,
        )

        images = generate_test_images(str(tmp_path), count=3)
        messages = build_vision_message([images[0]], "Опиши картинку")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Опиши картинку"
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]

    def test_build_vision_message_multiple(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            build_vision_message,
            generate_test_images,
        )

        images = generate_test_images(str(tmp_path), count=4)
        messages = build_vision_message([images[0], images[1], images[2]], "Опиши")

        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 4  # 1 text + 3 images
        assert content[0]["type"] == "text"
        for i in range(1, 4):
            assert content[i]["type"] == "image_url"

    def test_sawtooth_pattern(self):
        from llm_speed_benchmark.image_utils import sawtooth_image_count

        # max_images=3: 3, 2, 1, 2, 3, 2, 1, 2, 3...
        expected = [3, 2, 1, 2, 3, 2, 1, 2, 3]
        for i, exp in enumerate(expected):
            assert sawtooth_image_count(i, 3) == exp, f"call {i}: expected {exp}"

        # max_images=1: всегда 1
        for i in range(10):
            assert sawtooth_image_count(i, 1) == 1

        # max_images=5: 5,4,3,2,1,2,3,4,5,4,3,2,1...
        expected5 = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1]
        for i, exp in enumerate(expected5):
            assert sawtooth_image_count(i, 5) == exp, f"call {i}: expected {exp}"

    def test_discover_images(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            discover_images,
            generate_test_images,
        )

        generate_test_images(str(tmp_path), count=5)
        found = discover_images(tmp_path)
        assert len(found) == 5

    def test_discover_images_empty(self, tmp_path):
        from llm_speed_benchmark.image_utils import discover_images

        found = discover_images(tmp_path)
        assert found == []

    def test_discover_images_nonexistent(self):
        from llm_speed_benchmark.image_utils import discover_images

        found = discover_images("/nonexistent/path")
        assert found == []


class TestVisionWorker:
    """Тесты для _worker с моками."""

    @patch("llm_speed_benchmark.bench_vision.OpenAI")
    @patch("llm_speed_benchmark.bench_vision.StreamSession")
    def test_worker_sends_start_message(
        self, mock_session_cls, mock_openai_cls, tmp_path
    ):
        from multiprocessing import Event, Queue

        from llm_speed_benchmark.bench_vision import _worker
        from llm_speed_benchmark.image_utils import generate_test_images

        q = Queue()
        start_event = Event()
        start_event.set()

        images = generate_test_images(str(tmp_path), count=2)
        image_paths = [str(p) for p in images]

        # Мок StreamSession
        mock_session = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.completion_tokens = 100
        mock_metrics.prompt_tokens = 50
        mock_metrics.chunk_count = 50
        mock_metrics.elapsed = 1.0
        mock_metrics.ttft = 0.5
        mock_metrics.assistant_content = "Это тестовое изображение с геометрическими фигурами."
        mock_metrics.call_speed = 100.0
        mock_metrics.instant_speed = 100.0
        mock_session.run.return_value = mock_metrics
        mock_session.tokens_per_chunk = 2.0
        mock_session_cls.return_value = mock_session

        # Мок OpenAI client
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        p = __import__("threading", fromlist=["Thread"]).Thread(
            target=_worker,
            args=(0, q, start_event, image_paths, 2, ["Опиши"], 1, False, False, 1),
            daemon=True,
        )
        p.start()
        p.join(timeout=10)

        # Проверяем что воркер отправил start
        messages = []
        while not q.empty():
            try:
                messages.append(q.get_nowait())
            except Exception:
                break

        start_msgs = [m for m in messages if m.get("type") == "start"]
        assert len(start_msgs) == 1
        assert start_msgs[0]["id"] == 0
        assert start_msgs[0]["media"] == 4  # generate_test_images(min=4)

    @patch("llm_speed_benchmark.bench_vision.OpenAI")
    @patch("llm_speed_benchmark.bench_vision.StreamSession")
    def test_worker_sends_stats(
        self, mock_session_cls, mock_openai_cls, tmp_path
    ):
        from multiprocessing import Event, Queue

        from llm_speed_benchmark.bench_vision import _worker
        from llm_speed_benchmark.image_utils import generate_test_images

        q = Queue()
        start_event = Event()
        start_event.set()

        images = generate_test_images(str(tmp_path), count=1)
        image_paths = [str(p) for p in images]

        mock_session = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.completion_tokens = 80
        mock_metrics.prompt_tokens = 30
        mock_metrics.chunk_count = 40
        mock_metrics.elapsed = 0.8
        mock_metrics.ttft = 0.3
        mock_metrics.assistant_content = "На изображении градиент от красного к синему."
        mock_metrics.call_speed = 100.0
        mock_metrics.instant_speed = 100.0
        mock_session.run.return_value = mock_metrics
        mock_session.tokens_per_chunk = 2.0
        mock_session_cls.return_value = mock_session

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        p = __import__("threading", fromlist=["Thread"]).Thread(
            target=_worker,
            args=(1, q, start_event, image_paths, 2, ["Опиши"], 1, False, False, 1),
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


class TestVideoUtils:
    """Тесты для video utilities."""

    def test_build_video_message(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            build_video_message,
            generate_test_images,
        )

        # Создаём "видео" файл (фактически PNG с расширением .mp4 для теста)
        images = generate_test_images(str(tmp_path), count=1)
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(images[0].read_bytes())

        messages = build_video_message([fake_video], "Опиши видео")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Опиши видео"
        assert content[1]["type"] == "video_url"
        assert "data:video/mp4;base64," in content[1]["video_url"]["url"]

    def test_build_video_message_multiple(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            build_video_message,
            generate_test_images,
        )

        # Создаём 3 фейковых видео
        images = generate_test_images(str(tmp_path), count=4)
        videos = []
        for i in range(3):
            vid = tmp_path / f"video_{i}.mp4"
            vid.write_bytes(images[i].read_bytes())
            videos.append(str(vid))

        messages = build_video_message(videos, "Опиши")

        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 4  # 1 text + 3 videos
        assert content[0]["type"] == "text"
        for i in range(1, 4):
            assert content[i]["type"] == "video_url"

    def test_load_video_as_base64(self, tmp_path):
        import base64

        from llm_speed_benchmark.image_utils import (
            generate_test_images,
            load_video_as_base64,
        )

        images = generate_test_images(str(tmp_path), count=1)
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(images[0].read_bytes())

        mime, b64 = load_video_as_base64(fake_video)
        assert mime == "video/mp4"
        assert isinstance(b64, str)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_discover_videos(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            discover_videos,
            generate_test_images,
        )

        # Создаём фейковые видео из изображений (в отдельной папке)
        src_dir = tmp_path / "src"
        images = generate_test_images(str(src_dir), count=4)
        for i in range(3):
            vid = tmp_path / f"video_{i:03d}.mp4"
            vid.write_bytes(images[i].read_bytes())

        found = discover_videos(tmp_path)
        assert len(found) == 3

    def test_discover_videos_ignores_images(self, tmp_path):
        from llm_speed_benchmark.image_utils import (
            discover_videos,
            generate_test_images,
        )

        generate_test_images(str(tmp_path), count=5)
        found = discover_videos(tmp_path)
        assert found == []

    def test_discover_videos_empty(self, tmp_path):
        from llm_speed_benchmark.image_utils import discover_videos

        found = discover_videos(tmp_path)
        assert found == []

    def test_discover_videos_nonexistent(self):
        from llm_speed_benchmark.image_utils import discover_videos

        found = discover_videos("/nonexistent/path")
        assert found == []

    def test_ffmpeg_available(self):
        from llm_speed_benchmark.image_utils import _ffmpeg_available

        # Просто проверяем что функция не падает
        result = _ffmpeg_available()
        assert isinstance(result, bool)

    def test_generate_video_frames(self):
        from llm_speed_benchmark.image_utils import _generate_video_frame
        import random

        rng = random.Random(42)
        for variant in range(6):
            frame = _generate_video_frame(variant, (256, 256), 5, 15, rng)
            assert frame is not None
            assert frame.size == (256, 256)


class TestVisionLiveTable:
    """Тесты для VisionLiveTable."""

    def test_render_empty(self):
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        table = VisionLiveTable(None, 4)
        rendered = table.render()
        assert rendered is not None

    def test_render_with_workers(self):
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        table = VisionLiveTable(None, 2, response_width=40)
        table.mark_started(0, 10)
        table.mark_started(1, 10)

        table.update_stats({
            "id": 0,
            "calls": 5,
            "media": "test_image_000",
            "media_count": 1,
            "g": 500,
            "cg": 100,
            "speed": 50.0,
            "avg_speed": 45.0,
            "inst_speed": 55.0,
            "ttft": 0.5,
            "ttft_sum": 2.5,
            "tail": "На изображении представлены геометрические фигуры",
            "wall": "00:10",
        })

        rendered = table.render()
        assert rendered is not None

    def test_render_video_mode(self):
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        table = VisionLiveTable(None, 2, response_width=40, video_mode=True)
        table.mark_started(0, 5)
        table.mark_started(1, 5)

        table.update_stats({
            "id": 0,
            "calls": 3,
            "media": "test_video_000",
            "media_count": 1,
            "g": 300,
            "cg": 100,
            "speed": 40.0,
            "avg_speed": 35.0,
            "inst_speed": 45.0,
            "ttft": 1.2,
            "ttft_sum": 3.6,
            "tail": "На видео движущийся круг",
            "wall": "00:15",
        })

        rendered = table.render()
        assert rendered is not None

    def test_clean_tail_wide_chars(self):
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        # Wide CJK characters should be replaced
        result = VisionLiveTable._clean_tail("Hello 世界 Test", max_len=20)
        assert "世" not in result
        assert "界" not in result

    def test_clean_tail_truncation(self):
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        long_text = "A" * 100
        result = VisionLiveTable._clean_tail(long_text, max_len=20)
        assert len(result) <= 20
        assert result.endswith("...")

    def test_clean_tail_empty(self):
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        assert VisionLiveTable._clean_tail("", max_len=20) == ""


class TestSawtoothRegression:
    """Регрессионные тесты для sawtooth_image_count — баги которые реально ломали."""

    def test_zero_max_images_no_division_by_zero(self):
        """REGRESSION: max_images=0 вызывал ZeroDivisionError (elif вместо if)."""
        from llm_speed_benchmark.image_utils import sawtooth_image_count

        # Должен вернуть 1, а не упасть
        result = sawtooth_image_count(0, 0)
        assert result == 1

    def test_negative_max_images_no_crash(self):
        """max_images < 0 не должен крашиться."""
        from llm_speed_benchmark.image_utils import sawtooth_image_count

        result = sawtooth_image_count(5, -10)
        assert result == 1

    def test_max_images_2_pattern(self):
        """minимальный не-тривиальный паттерн: 2, 1, 2, 1..."""
        from llm_speed_benchmark.image_utils import sawtooth_image_count

        expected = [2, 1, 2, 1, 2, 1]
        for i, exp in enumerate(expected):
            assert sawtooth_image_count(i, 2) == exp, f"call {i}: expected {exp}"


class TestVisionStatsRegression:
    """Регрессионные тесты для VisionLiveTable — баги в маппинге ключей."""

    def test_update_stats_maps_g_to_gen(self):
        """REGRESSION: 'g' в update_stats не маппился на 'gen' -> итоги всегда 0."""
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        table = VisionLiveTable(None, 1)
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

    def test_update_stats_partial_does_not_lose_gen(self):
        """Частичное обновление не должно обнулить gen."""
        from llm_speed_benchmark.bench_vision import VisionLiveTable

        table = VisionLiveTable(None, 1)
        table.mark_started(0, 10)
        table.update_stats({"id": 0, "g": 1000, "calls": 1, "wall": "00:05", "tail": "ok"})
        # Второе обновление без 'g'
        table.update_stats({"id": 0, "calls": 2, "ttft": 0.3})
        assert table.workers[0]["gen"] == 1000  # не обнулён
        assert table.workers[0]["calls"] == 2


class TestVisionWorkerSkipErrors:
    """Worker: обработка ошибок с skip_errors."""

    @patch("llm_speed_benchmark.bench_vision.OpenAI")
    @patch("llm_speed_benchmark.bench_vision.StreamSession")
    def test_worker_continues_after_error_with_skip_errors(
        self, mock_session_cls, mock_openai_cls, tmp_path
    ):
        """С skip_errors=True воркер продолжает после ошибки."""
        from multiprocessing import Event, Queue

        from llm_speed_benchmark.bench_vision import _worker
        from llm_speed_benchmark.image_utils import generate_test_images

        q = Queue()
        start_event = Event()
        start_event.set()

        images = generate_test_images(str(tmp_path), count=2)
        image_paths = [str(p) for p in images]

        mock_session = MagicMock()
        call_count = [0]

        def run_side_effect(**kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("API error")
            metrics = MagicMock()
            metrics.completion_tokens = 100
            metrics.chunk_count = 50
            metrics.elapsed = 1.0
            metrics.ttft = 0.5
            metrics.assistant_content = "OK after error"
            metrics.call_speed = 100.0
            metrics.instant_speed = 100.0
            return metrics

        mock_session.run.side_effect = run_side_effect
        mock_session.tokens_per_chunk = 2.0
        mock_session_cls.return_value = mock_session

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        p = __import__("threading", fromlist=["Thread"]).Thread(
            target=_worker,
            args=(0, q, start_event, image_paths, 3, ["Опиши"], 1, True, False, 1),
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

        error_stops = [m for m in messages if m.get("type") == "error_stop"]
        # С skip_errors=True НЕ должно быть error_stop
        assert len(error_stops) == 0

        stats_msgs = [m for m in messages if m.get("type") == "stats"]
        # Должно быть >= 2 (ошибка + успешный вызов)
        assert len(stats_msgs) >= 2


class TestVisionWorkerVideoMode:
    """Worker: видео-режим."""

    @patch("llm_speed_benchmark.bench_vision.OpenAI")
    @patch("llm_speed_benchmark.bench_vision.StreamSession")
    def test_worker_video_mode_sends_video_message(
        self, mock_session_cls, mock_openai_cls, tmp_path
    ):
        """В video_mode воркер вызывает build_video_message."""
        from multiprocessing import Event, Queue

        from llm_speed_benchmark.bench_vision import _worker
        from llm_speed_benchmark.image_utils import generate_test_images

        q = Queue()
        start_event = Event()
        start_event.set()

        # Создаём фейковое видео
        images = generate_test_images(str(tmp_path), count=1)
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(images[0].read_bytes())
        video_paths = [str(fake_video)]

        mock_session = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.completion_tokens = 100
        mock_metrics.chunk_count = 50
        mock_metrics.elapsed = 1.0
        mock_metrics.ttft = 0.5
        mock_metrics.assistant_content = "Video description"
        mock_metrics.call_speed = 100.0
        mock_metrics.instant_speed = 100.0
        mock_session.run.return_value = mock_metrics
        mock_session.tokens_per_chunk = 2.0
        mock_session_cls.return_value = mock_session

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        p = __import__("threading", fromlist=["Thread"]).Thread(
            target=_worker,
            args=(0, q, start_event, video_paths, 2, ["Опиши"], 1, False, True, 1),
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


class TestVisionCli:
    """bench_vision CLI: парсинг аргументов."""

    def test_cli_default_args(self):
        from llm_speed_benchmark.bench_vision import cli

        with patch("sys.argv", ["bench_vision"]):
            with patch("llm_speed_benchmark.bench_vision.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["workers"] == 4
                assert call_kwargs["max_images"] == 1
                assert call_kwargs["max_videos"] == 1

    def test_cli_video_mode_args(self):
        from llm_speed_benchmark.bench_vision import cli

        with patch("sys.argv", ["bench_vision", "--max-videos", "3", "-w", "2"]):
            with patch("llm_speed_benchmark.bench_vision.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["max_videos"] == 3
                assert call_kwargs["workers"] == 2
                assert call_kwargs["video_mode"] is True

    def test_cli_skip_errors(self):
        from llm_speed_benchmark.bench_vision import cli

        with patch("sys.argv", ["bench_vision", "--skip-errors"]):
            with patch("llm_speed_benchmark.bench_vision.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["skip_errors"] is True

    def test_cli_custom_prompts(self):
        from llm_speed_benchmark.bench_vision import cli

        with patch("sys.argv", ["bench_vision", "-p", "Prompt 1", "-p", "Prompt 2"]):
            with patch("llm_speed_benchmark.bench_vision.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["prompts"] == ["Prompt 1", "Prompt 2"]

    def test_cli_max_images_zero_no_crash(self):
        """REGRESSION: --max-images 0 не должен крашиться."""
        from llm_speed_benchmark.bench_vision import cli

        with patch("sys.argv", ["bench_vision", "--max-images", "0", "--max-videos", "0"]):
            with patch("llm_speed_benchmark.bench_vision.run_benchmark") as mock_run:
                cli()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["max_images"] == 0
