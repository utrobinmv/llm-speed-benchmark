"""
llm_speed_benchmark/image_utils.py

Генерация и загрузка тестовых изображений для vision-бенчмарка.

Использует Pillow (PIL) для создания разнообразных изображений:
  - Градиенты, геометрические фигуры, паттерны, шум
  - Разрешения от 256x256 до 1024x768
"""

from __future__ import annotations

import base64
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter


def generate_test_images(
    output_dir: str | Path,
    count: int = 20,
    seed: int = 42,
) -> List[Path]:
    """Генерирует набор тестовых изображений для vision-бенчмарка.

    Создаёт разнообразные изображения: градиенты, фигуры, паттерны, шум.
    Сохраняет в PNG формате в указанную директорию.

    Args:
        output_dir: Директория для сохранения изображений.
        count: Количество изображений (минимум 4).
        seed: Seed для воспроизводимости.

    Returns:
        Список Path к сохранённым файлам.
    """
    if count < 4:
        count = 4

    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images: List[Path] = []

    for i in range(count):
        img = _generate_single_image(i, seed + i)
        path = output_dir / f"test_image_{i:03d}.png"
        img.save(str(path), "PNG")
        images.append(path)

    return images


def _generate_single_image(index: int, seed: int) -> "Image.Image":
    """Генерирует одно изображение по индексу.

    Типы изображений (циклически):
      0  — цветовой градиент (горизонтальный/вертикальный)
      1  — геометрические фигуры (круги, прямоугольники, треугольники)
      2  — паттерн (шахматная доска, полосы, точки)
      3  — случайный шум (Gaussian / uniform)
      4  — абстрактная композиция (множество фигур + blur)
      5  — текст на фоне
      6  — концентрические круги
      7  — радужный градиент
    """
    variant = index % 8

    # Разные разрешения для разнообразия
    sizes = [(512, 512), (256, 256), (1024, 768), (640, 480), (200, 300)]
    size = sizes[index % len(sizes)]

    rng = random.Random(seed)

    if variant == 0:
        return _gen_gradient(size, rng)
    elif variant == 1:
        return _gen_shapes(size, rng)
    elif variant == 2:
        return _gen_pattern(size, rng)
    elif variant == 3:
        return _gen_noise(size, rng)
    elif variant == 4:
        return _gen_abstract(size, rng)
    elif variant == 5:
        return _gen_text(size, rng)
    elif variant == 6:
        return _gen_concentric(size, rng)
    else:
        return _gen_rainbow(size, rng)


def _gen_gradient(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Цветовой градиент (R -> G -> B)."""
    w, h = size
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            r = int(255 * x / max(w - 1, 1))
            g = int(255 * y / max(h - 1, 1))
            b = 128
            pixels[x, y] = (r, g, b)
    return img


def _gen_shapes(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Геометрические фигуры на цветном фоне."""
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img)

    count = rng.randint(5, 15)
    for _ in range(count):
        shape_type = rng.choice(["circle", "rect", "polygon"])
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        x1 = rng.randint(0, size[0] // 2)
        y1 = rng.randint(0, size[1] // 2)
        x2 = x1 + rng.randint(30, size[0] // 3)
        y2 = y1 + rng.randint(30, size[1] // 3)

        if shape_type == "circle":
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
        elif shape_type == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
        else:
            pts = [(x1, y1), (x2, y1), ((x1 + x2) // 2, y2)]
            draw.polygon(pts, fill=color, outline=(0, 0, 0))

    return img


def _gen_pattern(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Паттерны: шахматная доска, полосы, точки."""
    pattern_type = rng.choice(["checker", "stripes_h", "stripes_v", "dots"])

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    cell = rng.randint(16, 64)

    if pattern_type == "checker":
        for x in range(0, size[0], cell):
            for y in range(0, size[1], cell):
                if ((x // cell) + (y // cell)) % 2 == 0:
                    draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(50, 50, 50))
    elif pattern_type == "stripes_h":
        for y in range(0, size[1], cell):
            if ((y // cell)) % 2 == 0:
                draw.rectangle([0, y, size[0] - 1, y + cell - 1], fill=(0, 100, 200))
    elif pattern_type == "stripes_v":
        for x in range(0, size[0], cell):
            if ((x // cell)) % 2 == 0:
                draw.rectangle([x, 0, x + cell - 1, size[1] - 1], fill=(0, 150, 100))
    else:  # dots
        for x in range(0, size[0], cell):
            for y in range(0, size[1], cell):
                if ((x // cell) + (y // cell)) % 2 == 0:
                    r = cell // 4
                    draw.ellipse([x + r, y + r, x + 3 * r, y + 3 * r], fill=(200, 50, 50))

    return img


def _gen_noise(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Случайный шум (uniform RGB)."""
    pixels = [rng.randint(0, 255) for _ in range(size[0] * size[1] * 3)]
    img = Image.frombytes("RGB", size, bytes(pixels))
    # Лёгкий blur чтобы не было чистого шума
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def _gen_abstract(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Абстрактная композиция: множество фигур + blur."""
    img = Image.new("RGB", size, (30, 30, 60))
    draw = ImageDraw.Draw(img)

    for _ in range(rng.randint(20, 50)):
        x1 = rng.randint(0, size[0])
        y1 = rng.randint(0, size[1])
        x2 = x1 + rng.randint(10, size[0] // 4)
        y2 = y1 + rng.randint(10, size[1] // 4)
        color = (
            rng.randint(50, 255),
            rng.randint(50, 255),
            rng.randint(50, 255),
        )
        if rng.random() > 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img.filter(ImageFilter.GaussianBlur(radius=rng.randint(3, 8)))


def _gen_text(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Текст на цветном фоне."""
    bg_color = (rng.randint(0, 100), rng.randint(0, 100), rng.randint(100, 255))
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    # Большой текст в центре
    text_color = (255, 255, 255)
    texts = ["HELLO", "BENCHMARK", "VISION", "TEST", "AI", "2025", "PILOT", "DATA"]
    text = rng.choice(texts)
    bbox = draw.textbbox((0, 0), text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size[0] - tw) // 2
    y = (size[1] - th) // 2
    draw.text((x, y), text, fill=text_color)

    return img


def _gen_concentric(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Концентрические круги."""
    img = Image.new("RGB", size, (20, 20, 40))
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2
    max_r = min(size) // 2
    step = max(10, max_r // 10)

    for r in range(max_r, 0, -step):
        hue = int(255 * r / max_r)
        color = (hue, 255 - hue, rng.randint(50, 200))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=step // 2)

    return img


def _gen_rainbow(size: Tuple[int, int], rng: random.Random) -> "Image.Image":
    """Радужный градиент (HSL-like)."""
    w, h = size
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(w):
        for y in range(h):
            angle = 2 * 3.14159 * (x / w + y / h) / 2
            r = int(128 + 127 * _sin(angle))
            g = int(128 + 127 * _sin(angle + 2.094))
            b = int(128 + 127 * _sin(angle + 4.189))
            pixels[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    return img


def _sin(x: float) -> float:
    """Простая аппроксимация sin без import math."""
    # Taylor series: sin(x) ~ x - x^3/6 + x^5/120
    x = x % (2 * 3.14159)
    return x - (x ** 3) / 6 + (x ** 5) / 120


def load_image_as_base64(image_path: str | Path) -> str:
    """Загружает изображение и кодирует в base64.

    Args:
        image_path: Путь к файлу изображения.

    Returns:
        Base64 строка PNG изображения.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_vision_message(
    image_paths: Sequence[str | Path],
    prompt: str = "Опиши подробно что изображено на этой картинке.",
) -> list:
    """Создаёт message для vision-запроса (OpenAI format).

    Args:
        image_paths: Путь или список путей к изображениям.
        prompt: Текстовый промпт.

    Returns:
        Список сообщений в формате OpenAI chat API.
    """
    content_parts: list = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        b64 = load_image_as_base64(img_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "auto",
            },
        })
    return [{"role": "user", "content": content_parts}]


def discover_images(directory: str | Path) -> List[Path]:
    """Находит все изображения в директории.

    Args:
        directory: Директория для поиска.

    Returns:
        Отсортированный список Path к изображениям (.png, .jpg, .jpeg, .webp).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    images = sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])
    return images


DEFAULT_PROMPTS = [
    "Опиши подробно что изображено на этой картинке.",
    "Расскажи что ты видишь на этом изображении.",
    "Опиши содержание этого изображения максимально подробно.",
    "Что нарисовано? Опиши цвета, формы и композицию.",
    "Подробно опиши визуальное содержание этого изображения.",
]


def sawtooth_image_count(call_index: int, max_images: int) -> int:
    """Пилообразный паттерн количества изображений.

    Последовательность: max, max-1, ..., 1, 2, ..., max-1, max, ...
    Период: 2 * (max_images - 1)

    Args:
        call_index: Номер вызова (0-based).
        max_images: Максимум изображений в одном запросе.

    Returns:
        Количество изображений для текущего вызова.
    """
    if max_images < 1:
        max_images = 1
    if max_images == 1:
        return 1

    period = 2 * (max_images - 1)
    phase = call_index % period
    if phase < max_images:
        return max_images - phase
    else:
        return phase - max_images + 2


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def generate_test_videos(
    output_dir: str | Path,
    count: int = 4,
    seed: int = 42,
    frames: int = 15,
    fps: int = 5,
    size: Tuple[int, int] = (256, 256),
) -> List[Path]:
    """Генерирует набор коротких тестовых видео для vision-бенчмарка.

    Создаёт анимированные MP4-видео с движущимися фигурами, градиентами
    и паттернами. Использует Pillow для генерации кадров и ffmpeg для
    кодирования в MP4 (libx264).

    Если ffmpeg недоступен, падает с ValueError — пользователь должен
    установить ffmpeg или указать свои видео через --videos.

    Args:
        output_dir: Директория для сохранения видео.
        count: Количество видео (минимум 1).
        seed: Seed для воспроизводимости.
        frames: Количество кадров в каждом видео.
        fps: Кадров в секунду.
        size: Разрешение видео (ширина, высота).

    Returns:
        Список Path к сохранённым .mp4 файлам.
    """
    if count < 1:
        count = 1

    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos: List[Path] = []

    for i in range(count):
        variant = i % 6
        rng = random.Random(seed + i)

        frames_list: List[Image.Image] = []
        for f in range(frames):
            frame = _generate_video_frame(variant, size, f, frames, rng)
            frames_list.append(frame)

        path = output_dir / f"test_video_{i:03d}.mp4"
        _frames_to_mp4(frames_list, str(path), fps)
        videos.append(path)

    return videos


def _generate_video_frame(
    variant: int,
    size: Tuple[int, int],
    frame_idx: int,
    total_frames: int,
    rng: random.Random,
) -> "Image.Image":
    """Генерирует один кадр видео.

    Варианты анимации:
      0 — движущийся круг
      1 — пульсирующие квадраты
      2 — вращающийся градиент
      3 — бегущие полосы
      4 — меняющийся шум
      5 — анимированный текст
    """
    t = frame_idx / max(total_frames - 1, 1)  # 0..1
    w, h = size

    if variant == 0:
        # Движущийся круг
        img = Image.new("RGB", size, (30, 30, 60))
        draw = ImageDraw.Draw(img)
        cx = int(size[0] * (0.2 + 0.6 * t))
        cy = int(size[1] * (0.5 + 0.3 * _sin(t * 6.28)))
        r = int(min(size) * 0.15)
        hue = int(255 * t)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(hue, 255 - hue, 128))
        return img

    elif variant == 1:
        # Пульсирующие квадраты
        img = Image.new("RGB", size, (20, 20, 40))
        draw = ImageDraw.Draw(img)
        for sq in range(5):
            offset = max(0, int(min(size) * (0.05 + 0.1 * _sin(t * 6.28 + sq))))
            color = (
                int(255 * ((sq + 1) / 6)),
                int(255 * (1 - (sq + 1) / 6)),
                128,
            )
            x0 = offset + sq * 10
            y0 = offset + sq * 10
            x1 = max(x0 + 1, size[0] - offset - sq * 10)
            y1 = max(y0 + 1, size[1] - offset - sq * 10)
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=color,
                width=2,
            )
        return img

    elif variant == 2:
        # Вращающийся градиент
        img = Image.new("RGB", size)
        pixels = img.load()  # type: ignore[assignment]
        angle = t * 6.28
        for x in range(w):
            for y in range(h):
                nx = (x - w // 2) / max(w, 1)
                ny = (y - h // 2) / max(h, 1)
                rot_x = nx * _cos(angle) - ny * _sin(angle)
                r = int(128 + 127 * rot_x)
                g = int(128 + 127 * ny)
                pixels[x, y] = (max(0, min(255, r)), max(0, min(255, g)), 100)  # type: ignore[index]
        return img

    elif variant == 3:
        # Бегущие горизонтальные полосы
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        stripe_h = 20
        offset = int(t * stripe_h)
        for y in range(0, h, stripe_h):
            if ((y + offset) // stripe_h) % 2 == 0:
                color = (0, int(100 + 155 * t), int(100 + 155 * (1 - t)))
                draw.rectangle([0, y, w - 1, min(y + stripe_h - 1, h - 1)], fill=color)
        return img

    elif variant == 4:
        # Меняющийся шум
        noise_rng = random.Random(frame_idx * 1000 + rng.randint(0, 10000))
        pixels = [noise_rng.randint(0, 255) for _ in range(w * h * 3)]
        img = Image.frombytes("RGB", size, bytes(pixels))
        return img.filter(ImageFilter.GaussianBlur(radius=3))

    else:
        # Анимированный текст (появляется по буквам)
        bg_color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(100, 255))
        img = Image.new("RGB", size, bg_color)
        draw = ImageDraw.Draw(img)
        text = "BENCHMARK"
        visible_chars = max(1, int(len(text) * t))
        partial = text[:visible_chars]
        bbox = draw.textbbox((0, 0), partial)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (size[0] - tw) // 2
        y = (size[1] - th) // 2
        draw.text((x, y), partial, fill=(255, 255, 255))
        return img


def _cos(x: float) -> float:
    """cos(x) через sin(x + pi/2)."""
    return _sin(x + 1.5708)


def _frames_to_mp4(frames: List["Image.Image"], output_path: str, fps: int) -> None:
    """Кодирует список PIL-кадров в MP4 через ffmpeg.

    Args:
        frames: Список PIL Image.
        output_path: Путь к выходному .mp4.
        fps: Кадров в секунду.
    """
    if not _ffmpeg_available():
        raise ValueError(
            "ffmpeg не найден. Установите ffmpeg или укажите свои видео через --videos."
        )

    width, height = frames[0].size
    with tempfile.TemporaryDirectory() as tmpdir:
        # Сохраняем кадры как PNG
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                output_path,
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )


def _ffmpeg_available() -> bool:
    """Проверяет доступность ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def load_video_as_base64(video_path: str | Path) -> Tuple[str, str]:
    """Загружает видео и кодирует в base64.

    Args:
        video_path: Путь к файлу видео.

    Returns:
        Кортеж (mime_type, base64_string).
    """
    suffix = Path(video_path).suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".gif": "image/gif",
    }
    mime = mime_map.get(suffix, "video/mp4")

    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return mime, b64


def build_video_message(
    video_paths: Sequence[str | Path],
    prompt: str = "Опиши подробно что происходит на этом видео.",
) -> list:
    """Создаёт message для video-запроса (OpenAI format).

    Args:
        video_paths: Путь или список путей к видео.
        prompt: Текстовый промпт.

    Returns:
        Список сообщений в формате OpenAI chat API.
    """
    content_parts: list = [{"type": "text", "text": prompt}]
    for vid_path in video_paths:
        mime, b64 = load_video_as_base64(vid_path)
        content_parts.append({
            "type": "video_url",
            "video_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": "auto",
            },
        })
    return [{"role": "user", "content": content_parts}]


def discover_videos(directory: str | Path) -> List[Path]:
    """Находит все видео в директории.

    Args:
        directory: Директория для поиска.

    Returns:
        Отсортированный список Path к видео (.mp4, .mov, .avi, .webm, .gif).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    extensions = {".mp4", ".mov", ".avi", ".webm", ".gif"}
    videos = sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])
    return videos


DEFAULT_VIDEO_PROMPTS = [
    "Опиши подробно что происходит на этом видео.",
    "Расскажи что ты видишь в этом видео.",
    "Опиши действия и события на видео максимально подробно.",
    "Что происходит? Опиши движение, цвета и изменения.",
    "Подробно опиши содержание этого видео.",
]
