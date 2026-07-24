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
            args=(0, q, start_event, image_paths, 2, ["Опиши"], 1),
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
            args=(1, q, start_event, image_paths, 2, ["Опиши"], 1),
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
            "img": "test_image_000",
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
