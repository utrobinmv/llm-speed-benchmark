# AGENTS.md — Инструкции для AI-агента

## Контекст проекта

**llm-speed-benchmark** — pip-устанавливаемый пакет для бенчмарка скорости стриминга LLM через OpenAI-compatible API (vLLM, Ollama и др.).

Три режима:
- `bench_single` — последовательные вызовы, накопление контекста, статистика
- `bench_multi` — N параллельных процессов (multiprocessing), Rich Live-таблица
- `bench_vision` — бенчмарк vision/video-модели: отправляет изображения или видео, измеряет TTFT и скорость генерации
- `bench_audio` — бенчмарк аудио-модели: отправляет аудио файлы, измеряет TTFT и скорость транскрипции

## Установка

```bash
# Из Git
pip install git+https://github.com/utrobinmv/llm-speed-benchmark.git

# Локально (editable)
cd ~/workspace/projects/llm-speed-benchmark
source .venv
pip install -e ".[dev]"
```

## Структура пакета

```
llm-speed-benchmark/
├── pyproject.toml                   # setuptools, entry points, deps
├── src/llm_speed_benchmark/         # ИСХОДНЫЙ КОД ПАКЕТА
│   ├── __init__.py                  # __version__
│   ├── bench_single.py              # Одиночный воркер + cli()
│   ├── bench_multi.py               # Многопроцессный + LiveTable + cli()
│   ├── bench_vision.py              # Vision-бенчмарк + cli()
│   ├── bench_audio.py               # Audio-бенчмарк + cli()
│   ├── image_utils.py               # Генерация/загрузка изображений
│   ├── audio_utils.py               # Загрузка аудио из бандла (assets/audio/)
│   ├── streaming.py                 # StreamSession + StreamMetrics
│   ├── cli_common.py                # add_common_args(), apply_config()
│   └── utils.py                     # get_client(), truncate_history(), progress_bar(), format_time()
├── assets/audio/                    # 10 WAV-файлов с русской речью (~1.7 МБ)
├── tests/
│   ├── __init__.py
│   ├── test_bench_single.py         # pytest (mock OpenAI)
│   ├── test_bench_multi.py          # pytest (mock OpenAI + multiprocessing)
│   ├── test_bench_vision.py         # pytest (image_utils + vision worker)
│   ├── test_bench_audio.py          # pytest (audio_utils + audio worker)
│   ├── test_cli_common.py           # pytest (CLI args)
│   ├── test_streaming.py            # pytest (StreamSession)
│   └── test_long_context.py         # pytest (long context dataset)
├── .venv                            # source .venv -> активация
├── .env                             # BASE_URL, API_KEY, MODEL, MAX_CONTEXT_TOKENS
├── INSTALL.md
├── README.md
└── AGENTS.md
```

## Entry points (pyproject.toml)

```toml
[project.scripts]
bench_single = "llm_speed_benchmark.bench_single:cli"
bench_multi = "llm_speed_benchmark.bench_multi:cli"
bench_vision = "llm_speed_benchmark.bench_vision:cli"
bench_audio = "llm_speed_benchmark.bench_audio:cli"
```

## Зависимости

| Пакет | Зачем |
|---|---|
| `openai>=1.0.0` | OpenAI-compatible HTTP клиент |
| `python-dotenv>=1.0.0` | Загрузка `.env` |
| `rich>=13.0.0` | Rich Live-таблица в `bench_multi` и `bench_vision` |
| `Pillow>=10.0.0` | Генерация тестовых изображений для `bench_vision` |

Dev: `ruff`, `mypy`, `pytest`.

## Конфигурация (.env)

```ini
BASE_URL=http://localhost:8000/v1
API_KEY=sk-vllm-qwen3.5-0.8b
MODEL=qwen3.5-0.8b
MAX_CONTEXT_TOKENS=262144
```

Загрузка: `utils.py` → `load_dotenv()` из cwd + `~/.llm-speed-benchmark.env`.

## CLI

Приоритет: **CLI аргументы > переменные окружения > `.env`**

```bash
# bench_single
bench_single                                          # До лимита контекста
bench_single --duration 60                            # 60 секунд
bench_single -u http://localhost:8000/v1 -m qwen3.5-0.8b
bench_single -u http://10.0.0.5:8000/v1 -m llama-3.1-8b --duration 120

# bench_multi
bench_multi                                           # 4 воркера, до Ctrl+C
bench_multi -w 8 -d 60                                # 8 воркеров, 60 сек
bench_multi --response-width 120                      # Широкая колонка
bench_multi --max-context 32768                       # Переопределение лимита
# bench_multi -u http://localhost:8000/v1 -m qwen3.5-0.8b -w 4 -d 60

# bench_vision
bench_vision                                            # 4 воркера, авто-генерация изображений
bench_vision -w 8 -d 120                                # 8 воркеров, 120 сек
bench_vision --images ~/workspace/data/benchmark_images/ # свои изображения
bench_vision -p "Опиши что на картинке" -p "Что ты видишь?"  # кастомные промпты
bench_vision --max-images 4                             # макс 4 изображения в запросе
bench_vision --videos ~/workspace/data/test_videos/     # видео-режим: свои видео
bench_vision --max-videos 4                             # видео-режим, макс 4 в запросе (авто-генерация)
bench_vision --max-videos 4 -w 8 -d 60                  # видео, 8 воркеров, 60 сек

# bench_audio
bench_audio                                             # 4 воркера, авто-генерация аудио
bench_audio -w 8 -d 120                                 # 8 воркеров, 120 сек
bench_audio --audio ~/workspace/data/audio/             # свои аудио (.wav, .mp3)
bench_audio -p "Транскрибируй" -p "Распознай текст"     # кастомные промпты
bench_audio --max-audio 1                               # макс 1 аудио в запросе

 # Общие аргументы
-u, --base-url    Адрес API (OpenAI-compatible)
-k, --api-key     API ключ
-m, --model       Название модели
--max-context     Переопределение MAX_CONTEXT_TOKENS
```

## Важные замечания

1. **vLLM сервер** должен быть запущен на `BASE_URL` перед запуском
2. **multiprocessing** — каждый воркер создаёт свой OpenAI клиент
3. **Ctrl+C** — корректно останавливает все процессы
4. **src/ layout** — код пакета в `src/llm_speed_benchmark/`, НЕ в корне
5. **Тесты**: `pytest` (моки OpenAI, не требуют сервера)
