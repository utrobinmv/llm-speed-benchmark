# LLM Speed Benchmark

Бенчмарк скорости стриминга LLM через OpenAI-compatible API (vLLM, Ollama, и любой совместимый сервер).

Шесть режимов: **однопоточный**, **многопроцессный** с Rich Live-таблицей, **vision** для мультимодальных моделей, **audio** для моделей с поддержкой аудио, **transcription** для ASR-моделей, **TTS** для синтеза речи, и **embedding** для эмбединг-моделей.

## Установка

### Из Git-репозитория

```bash
pip install git+https://github.com/utrobinmv/llm-speed-benchmark.git
```

### Из локальной копии

```bash
git clone https://github.com/utrobinmv/llm-speed-benchmark.git
cd llm-speed-benchmark
pip install -e .
```

### С dev-зависимостями

```bash
pip install -e ".[dev]"
```

После установки в PATH появляются команды `bench_single`, `bench_multi`, `bench_vision` и `bench_audio`.

## Настройка

Параметры можно задать **тремя способами** (приоритет сверху вниз):

1. **Аргументы командной строки** (высший приоритет)
2. **Переменные окружения** (`BASE_URL`, `API_KEY`, `MODEL`)
3. **Файл `.env`** в рабочей директории или `~/.llm-speed-benchmark.env`

```ini
BASE_URL=http://localhost:8000/v1
API_KEY=sk-vllm-qwen3.5-0.8b
MODEL=qwen3.5-0.8b
MAX_CONTEXT_TOKENS=262144
```

| Параметр | Описание |
|---|---|
| `BASE_URL` | Адрес LLM-сервера (OpenAI-compatible API) |
| `API_KEY` | Ключ (vLLM не проверяет, можно любой) |
| `MODEL` | Название модели |
| `MAX_CONTEXT_TOKENS` | Лимит контекста (85% — триггер обрезки истории) |

## Использование

### Одиночный воркер (`bench_single`)

Последовательные вызовы с накоплением контекста:

```bash
bench_single                          # Бежит до лимита контекста (из .env)
bench_single --duration 60            # Ограничение по времени (60 сек)
bench_single -u http://localhost:8000/v1 -m qwen3.5-0.8b
bench_single -u http://10.0.0.5:8000/v1 -m llama-3.1-8b --duration 120
```

**Метрики:**
- Prompt tokens / Gen tokens / Total tokens
- Скорость генерации (avg и instant t/s)
- TTFT (Time To First Token)
- Progress bar заполнения контекста

### Многопроцессный (`bench_multi`)

N параллельных процессов с Rich Live-таблицей:

```bash
bench_multi                                    # 4 воркера, до Ctrl+C
bench_multi -w 8                               # 8 воркеров
bench_multi -w 4 -d 60                         # 4 воркера, 60 секунд
bench_multi -w 4 -d 60 --response-width 120    # Широкая колонка ответа
bench_multi -w 4 --max-context 32768           # Переопределение лимита контекста
bench_multi -u http://localhost:8000/v1 -m qwen3.5-0.8b -w 4 -d 60
```

**Особенности:**
- Каждый воркер — независимый процесс с уникальным KV-кэшем
- При достижении лимита контекста — автоматический новый раунд
- Live-обновление метрик во время стриминга + wall-time каждую секунду
- Остановка: `Ctrl+C`

**Метрики таблицы:**

| Колонка | Описание |
|---|---|
| Gen | Точные токены генерации (из API usage) |
| Gen est | Оценка: Gen + чанки текущего вызова (1 чанк ~ 1 токен; после стрима = Gen) |
| Chunks | Всего чанков с начала воркера |
| CallGen | Токены в последнем вызове |
| Speed | Чанки / время вызова (мгновенная скорость) |
| Avg | Gen est / время работы воркера |
| TTFT | Time To First Token последнего вызова |
| TTFT sum | Суммарный TTFT всех вызовов |

### Vision-бенчмарк (`bench_vision`)

Тестирует скорость мультимодальных моделей (LLM с поддержкой изображений и видео).

**Три режима:**

1. **Image-only** — только изображения (колонки `Imgs | Image`)
2. **Video-only** — только видео (колонки `Vid | Video`)
3. **Mixed** — и изображения, и видео в одном запросе (колонки `Imgs | Image | Vid | Video`)

```bash
# Image-only (по умолчанию)
bench_vision                                       # 4 воркера, авто-генерация изображений
bench_vision -w 8 -d 120                           # 8 воркеров, 120 сек
bench_vision --images ~/workspace/data/images/     # свои изображения (.png, .jpg, .webp)
bench_vision --max-images 4                        # макс 4 изображения в запросе
bench_vision -p "Опиши что на картинке"            # кастомный промпт

# Video-only
bench_vision --videos ~/workspace/data/videos/     # свои видео (.mp4, .mov, .avi, .webm)
bench_vision --max-videos 4 --max-images 0         # макс 4 видео в запросе
bench_vision --max-videos 4 -w 8 -d 60            # видео, 8 воркеров, 60 сек

# Mixed (оба типа в одном запросе)
bench_vision --max-videos 2 --max-images 3         # 2 видео + 3 изображения в запросе
bench_vision --max-videos 3 --max-images 4 -w 8 -d 60  # mixed, 8 воркеров
```

**Особенности:**
- Три режима: image-only, video-only, mixed (оба типа в одном запросе)
- Каждый воркер циклически проходит по всем медиа
- Разные промпты для каждого запроса (ротация)
- Пилообразный паттерн: `max, max-1, ..., 1, 2, ..., max-1` (независимо для изображений и видео)
- В mixed-режиме в Rich-таблице отображаются две медиа-колонки: `Imgs | Image | Vid | Video`
- Медиа кодируются в base64 и отправляются через OpenAI vision API (`image_url` и `video_url` с `data:` URI)
- Если медиа не найдены — автоматически генерируются (изображения: градиенты, фигуры, паттерны, шум; видео: анимированные фигуры через ffmpeg)
- Те же метрики: TTFT, скорость генерации, мгновенная скорость

**Аргументы bench_vision:**

| Аргумент | Краткий | Описание |
|---|---|---|
| `--workers` | `-w` | Количество воркеров (по умолчанию: 4) |
| `--duration` | `-d` | Длительность в секундах |
| `--images` | | Директория с изображениями |
| `--videos` | | Директория с видео (если >0 вместе с --max-images>0 — mixed режим) |
| `--max-images` | | Максимум изображений в запросе (пилообразный паттерн; 0 = без изображений) |
| `--max-videos` | | Максимум видео в запросе (пилообразный паттерн; 0 = без видео) |
| `--prompt` | `-p` | Кастомный промпт (можно несколько) |
| `--response-width` | | Ширина колонки Response |
| `--skip-errors` | | Продолжать после ошибки |

### Audio-бенчмарк (`bench_audio`)

Для моделей с поддержкой аудио/транскрипции:

```bash
bench_audio                                       # 4 воркера, бандл из 10 WAV с речью
bench_audio -w 8 -d 120                           # 8 воркеров, 120 сек
bench_audio --audio ~/workspace/data/audio/       # свои аудио (.wav, .mp3)
bench_audio --max-audio 1                         # макс 1 аудио в запросе
bench_audio -p "Транскрибируй это аудио"           # кастомный промпт
bench_audio -p "Транскрибируй" -p "Распознай текст" # несколько промптов
```

**Особенности:**
- Каждый воркер циклически проходит по всем аудио файлам
- Разные промпты для каждого запроса (ротация)
- Аудио кодируются в base64 и отправляются через OpenAI audio API (`audio_url` с `data:` URI)
- В комплекте 10 WAV-файлов с русской речью (~1.7 МБ)
- Те же метрики: TTFT, скорость генерации, мгновенная скорость

**Аргументы bench_audio:**

| Аргумент | Краткий | Описание |
|---|---|---|
| `--workers` | `-w` | Количество воркеров (по умолчанию: 4) |
| `--duration` | `-d` | Длительность в секундах |
| `--audio` | | Директория с аудио файлами |
| `--max-audio` | | Максимум аудио в запросе |
| `--prompt` | `-p` | Кастомный промпт (можно несколько) |
| `--response-width` | | Ширина колонки Response |
| `--skip-errors` | | Продолжать после ошибки |

### Transcription-бенчмарк (`bench_transcription`)

Для ASR-моделей (Whisper, Qwen-ASR) — транскрипция речи в текст:

```bash
bench_transcription                                   # 4 воркера, бандл аудио
bench_transcription -w 2 -d 120                       # 2 воркера, 120 сек
bench_transcription --audio ~/workspace/data/audio/   # свои аудио
bench_transcription --durations 5 10 30 60            # генерация аудио разной длины
bench_transcription --max-duration 30                 # фильтр по максимальной длине
bench_transcription --language ru                     # язык транскрипции
bench_transcription --mode once                       # один проход (не зацикливание)
bench_transcription --model-timeout 1200              # таймаут для долгих аудио
```

**Особенности:**
- Endpoint: `audio.transcriptions` (не `chat.completions`)
- Генерация аудио заданной длительности через зацикливание бандла
- Метрики: RTF (Real-Time Factor), токены/сек, TTFT
- Конфиг из `.env`: `TRANSCRIPTION_BASE_URL`, `TRANSCRIPTION_API_KEY`, `TRANSCRIPTION_MODEL`

**Аргументы bench_transcription:**

| Аргумент | Краткий | Описание |
|---|---|---|
| `--workers` | `-w` | Количество воркеров (по умолчанию: 4) |
| `--duration` | `-d` | Длительность в секундах |
| `--audio` | | Директория с аудио файлами |
| `--durations` | | Список длительностей аудио для генерации (сек) |
| `--max-duration` | | Максимальная длительность аудио (фильтр) |
| `--model-timeout` | | Таймаут запроса к модели |
| `--prompt` | `-p` | Промпт для контекста |
| `--language` | `-l` | Язык транскрипции |
| `--response-format` | | Формат ответа (json, text, srt, verbose_json, vtt) |
| `--response-width` | | Ширина колонки Transcript |
| `--skip-errors` | | Продолжать после ошибки |
| `--mode` | | `cycling` (зацикливание) или `once` (один проход) |

### TTS-бенчмарк (`bench_tts`)

Для моделей синтеза речи (Qwen3-TTS и другие):

```bash
bench_tts                                           # 4 воркера, тексты разной длины
bench_tts -w 2 -d 120                               # 2 воркера, 120 сек
bench_tts --char-counts 10 50 200 500               # тексты указанной длины (символы)
bench_tts --voice agata                             # голос TTS
bench_tts --response-format wav                     # формат аудио (wav, mp3, opus, pcm)
bench_tts --mode once                               # один проход
bench_tts --model-timeout 600                       # таймаут запроса
```

**Особенности:**
- Endpoint: `audio.speech` (POST /v1/audio/speech)
- Тексты разной длины: short (10-50), medium (50-200), long (200-500), very_long (500+)
- Метрики: TTFT, char/s (символы/сек), KB/s (размер аудио/сек), avg processing time
- Конфиг из `.env`: `TTS_BASE_URL`, `TTS_API_KEY`, `TTS_MODEL`

**Аргументы bench_tts:**

| Аргумент | Краткий | Описание |
|---|---|---|
| `--workers` | `-w` | Количество воркеров (по умолчанию: 4) |
| `--duration` | `-d` | Длительность в секундах |
| `--char-counts` | | Список длин текстов в символах |
| `--voice` | | Голос TTS (по умолчанию: agata) |
| `--response-format` | | Формат аудио (wav, mp3, opus, pcm) |
| `--model-timeout` | | Таймаут запроса к модели |
| `--response-width` | | Ширина колонки Text |
| `--skip-errors` | | Продолжать после ошибки |
| `--mode` | | `cycling` (зацикливание) или `once` (один проход) |

### Embedding-бенчмарк (`bench_emb`)

Для эмбединг-моделей (embeddinggemma-300m и другие):

```bash
bench_emb                                           # 4 воркера, тексты разной длины
bench_emb -w 8 -d 120                               # 8 воркеров, 120 сек
bench_emb --char-counts 10 50 200 500               # тексты указанной длины (символы)
bench_emb --batch-sizes 1 5 10                      # batch-запросы (несколько текстов в одном вызове)
bench_emb --mode once                               # один проход
bench_emb --model-timeout 600                       # таймаут запроса
```

**Особенности:**
- Endpoint: `embeddings` (POST /v1/embeddings)
- Поддержка batch-запросов: несколько текстов в одном API-вызове
- Метрики: TTFT, char/s (символы/сек), doc/s (документы/сек), размерность эмбединга
- Конфиг из `.env`: `EMB_BASE_URL`, `EMB_API_KEY`, `EMB_MODEL`

**Аргументы bench_emb:**

| Аргумент | Краткий | Описание |
|---|---|---|
| `--workers` | `-w` | Количество воркеров (по умолчанию: 4) |
| `--duration` | `-d` | Длительность в секундах |
| `--char-counts` | | Список длин текстов в символах |
| `--batch-sizes` | | Размеры батча (1, 5, 10 и т.д.) |
| `--model-timeout` | | Таймаут запроса к модели |
| `--response-width` | | Ширина колонки Input |
| `--skip-errors` | | Продолжать после ошибки |
| `--mode` | | `cycling` (зацикливание) или `once` (один проход) |

### Общие аргументы CLI

| Аргумент | Краткий | Описание |
|---|---|---|
| `--base-url` | `-u` | Адрес API (OpenAI-compatible) |
| `--api-key` | `-k` | API ключ |
| `--model` | `-m` | Название модели |
| `--max-context` | | Переопределение MAX_CONTEXT_TOKENS |
| `--duration` | `-d` | Длительность в секундах (только bench_multi — `-d`, bench_single — `--duration`) |
| `--workers` | `-w` | Количество воркеров (только bench_multi) |
| `--response-width` | | Ширина колонки Response (только bench_multi) |

## Архитектура bench_multi

```
┌──────────────────────────────────────────────┐
│              Main Process                     │
│  display_loop() → Queue → Rich Live table    │
└──────┬───────────────────────────────────────┘
       │ multiprocessing.Queue
       │
┌──────┼──────┬───────┬──────┐
▼      ▼      ▼       ▼      ▼
Worker 0  Worker 1  ...  Worker N-1
```

## Структура проекта

```
llm-speed-benchmark/
├── pyproject.toml                   # Конфигурация пакета (setuptools)
├── src/llm_speed_benchmark/         # Исходный код пакета
│   ├── __init__.py
│   ├── bench_single.py              # Одиночный воркер
│   ├── bench_multi.py               # Многопроцессный воркер
│   ├── bench_vision.py              # Vision-бенчмарк
│   ├── bench_audio.py               # Audio-бенчмарк (chat.completions)
│   ├── bench_transcription.py       # ASR-бенчмарк (audio.transcriptions)
│   ├── bench_tts.py                 # TTS-бенчмарк (audio.speech)
│   ├── bench_emb.py                 # Embedding-бенчмарк (embeddings)
│   ├── live_table.py                # BaseLiveTable — общая Live-таблица
│   ├── worker_common.py             # Общие воркер-хелперы
│   ├── image_utils.py               # Генерация/загрузка изображений
│   ├── audio_utils.py               # Загрузка аудио из бандла
│   ├── transcription_utils.py       # Генерация аудио разной длины для ASR
│   ├── streaming.py                 # StreamSession + StreamMetrics
│   ├── cli_common.py                # Общие CLI аргументы
│   └── utils.py                     # Общие утилиты
├── assets/audio/                    # Тестовые WAV-файлы с речью (10 шт)
├── tests/                           # Тесты (pytest)
│   ├── __init__.py
│   ├── test_bench_single.py
│   ├── test_bench_multi.py
│   ├── test_bench_vision.py
│   ├── test_bench_audio.py
│   ├── test_bench_transcription.py
│   ├── test_bench_tts.py
│   ├── test_bench_emb.py
│   ├── test_cli_common.py
│   ├── test_streaming.py
│   └── test_long_context.py
├── .env                             # Конфигурация (BASE_URL, API_KEY, MODEL + TRANSCRIPTION_*, TTS_*, EMB_*)
├── INSTALL.md                       # Инструкция по установке
├── README.md
└── AGENTS.md
```

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

## Сборка дистрибутива

```bash
pip install build
python -m build
# → dist/llm_speed_benchmark-<version>-py3-none-any.whl
```

## Лицензия

MIT
