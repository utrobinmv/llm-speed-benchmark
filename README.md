# LLM Speed Benchmark

Бенчмарк скорости стриминга LLM через OpenAI-compatible API (vLLM, Ollama, и любой совместимый сервер).

Четыре режима: **однопоточный**, **многопроцессный** с Rich Live-таблицей, **vision** для мультимодальных моделей, и **audio** для моделей с поддержкой аудио/транскрипции.

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
- Live-обновление метрик каждые 0.5 сек
- Остановка: `Ctrl+C`

### Vision-бенчмарк (`bench_vision`)

Тестирует скорость мультимодальных моделей (LLM с поддержкой изображений и видео):

```bash
bench_vision                                       # 4 воркера, авто-генерация изображений
bench_vision -w 8 -d 120                           # 8 воркеров, 120 сек
bench_vision --images ~/workspace/data/images/     # свои изображения (.png, .jpg, .webp)
bench_vision --max-images 4                        # макс 4 изображения в запросе
bench_vision -p "Опиши что на картинке"            # кастомный промпт
bench_vision -p "Опиши" -p "Что ты видишь?"       # несколько промптов
```

**Видео-режим** (требует ffmpeg для авто-генерации):

```bash
bench_vision --videos ~/workspace/data/videos/     # свои видео (.mp4, .mov, .avi, .webm)
bench_vision --max-videos 4                        # макс 4 видео в запросе (авто-генерация)
bench_vision --max-videos 4 -w 8 -d 60            # видео, 8 воркеров, 60 сек
```

**Особенности:**
- Каждый воркер циклически проходит по всем медиа
- Разные промпты для каждого запроса (ротация)
- Медиа кодируются в base64 и отправляются через OpenAI vision API
- Если медиа не найдены — автоматически генерируются (градиенты, фигуры, паттерны, шум)
- Пилообразный паттерн: `max, max-1, ..., 1, 2, ..., max-1`
- Те же метрики: TTFT, скорость генерации, мгновенная скорость

**Аргументы bench_vision:**

| Аргумент | Краткий | Описание |
|---|---|---|
| `--workers` | `-w` | Количество воркеров (по умолчанию: 4) |
| `--duration` | `-d` | Длительность в секундах |
| `--images` | | Директория с изображениями |
| `--videos` | | Директория с видео (переключает в видео-режим) |
| `--max-images` | | Максимум изображений в запросе (пилообразный паттерн) |
| `--max-videos` | | Максимум видео в запросе (пилообразный паттерн) |
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
│   ├── bench_audio.py               # Audio-бенчмарк
│   ├── image_utils.py               # Генерация/загрузка изображений
│   ├── audio_utils.py               # Загрузка аудио из бандла
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
│   ├── test_cli_common.py
│   ├── test_streaming.py
│   └── test_long_context.py
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
