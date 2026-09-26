# AGENTS.md — Инструкции для AI-агента

## Контекст проекта

**llm-speed-benchmark** — pip-устанавливаемый пакет для бенчмарка скорости стриминга LLM через OpenAI-compatible API (vLLM, Ollama и др.).

Семь режимов:
- `bench_single` — последовательные вызовы, накопление контекста, статистика
- `bench_multi` — N параллельных процессов (multiprocessing), Rich Live-таблица
- `bench_vision` — бенчмарк мультимодальной модели: изображения, видео или И то и другое в одном запросе
- `bench_audio` — бенчмарк аудио-модели: отправляет аудио файлы через chat.completions, измеряет TTFT и скорость
- `bench_transcription` — бенчмарк ASR-моделей (Whisper, Qwen-ASR): endpoint audio.transcriptions, RTF, токены/сек
- `bench_tts` — бенчмарк TTS-моделей (Qwen3-TTS): endpoint audio.speech, char/s, KB/s
- `bench_emb` — бенчмарк эмбединг-моделей (embeddinggemma-300m): endpoint embeddings, char/s, doc/s, размерность

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
│   ├── bench_vision.py              # Vision/Video/Mixed бенчмарк + cli()
│   ├── bench_audio.py               # Audio-бенчмарк (chat.completions) + cli()
│   ├── bench_transcription.py       # ASR-бенчмарк (audio.transcriptions) + cli()
│   ├── bench_tts.py                 # TTS-бенчмарк (audio.speech) + cli()
│   ├── bench_emb.py                 # Embedding-бенчмарк (embeddings) + cli()
│   ├── live_table.py                # BaseLiveTable — общий класс Live-таблицы
│   ├── worker_common.py             # Общие воркер-хелперы (time_sender, on_chunk, stats)
│   ├── image_utils.py               # Генерация/загрузка изображений, видео, mixed-message
│   ├── audio_utils.py               # Загрузка аудио из бандла (assets/audio/)
│   ├── transcription_utils.py       # Генерация аудио разной длины для ASR-бенчмарка
│   ├── streaming.py                 # StreamSession + StreamMetrics
│   ├── cli_common.py                # add_common_args(), apply_config()
│   └── utils.py                     # get_client(), truncate_history(), progress_bar(), format_time()
├── assets/audio/                    # 10 WAV-файлов с русской речью (~1.7 МБ)
├── tests/
│   ├── __init__.py
│   ├── test_bench_single.py
│   ├── test_bench_multi.py
│   ├── test_bench_vision.py         # Тесты: image_utils, worker, LiveTable, CLI
│   ├── test_bench_audio.py
│   ├── test_bench_transcription.py  # Тесты: transcription_utils, worker, LiveTable, CLI
│   ├── test_bench_tts.py            # Тесты: build_text_tasks, TTSLiveTable, CLI, конфиг
│   ├── test_bench_emb.py            # Тесты: build_emb_tasks, EmbeddingLiveTable, CLI, конфиг
│   ├── test_cli_common.py
│   ├── test_streaming.py
│   └── test_long_context.py
├── .venv                            # source .venv -> активация
├── .env                             # BASE_URL, API_KEY, MODEL + TRANSCRIPTION_*, TTS_*, EMB_*
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
bench_transcription = "llm_speed_benchmark.bench_transcription:cli"
bench_tts = "llm_speed_benchmark.bench_tts:cli"
bench_emb = "llm_speed_benchmark.bench_emb:cli"
```

## bench_vision — три режима работы

### 1. Image-only (max_videos=0, max_images>0)
Колонки: `Imgs | Image`
```
bench_vision                          # дефолт: 4 воркера, макс 1 изобр/запрос
bench_vision --max-images 4
```

### 2. Video-only (max_videos>0, max_images=0)
Колонки: `Vid | Video`
```
bench_vision --max-videos 4 --max-images 0
```

### 3. Mixed mode (max_videos>0 И max_images>0)
Колонки: `Imgs | Image | Vid | Video` — обе пары одновременно
```
bench_vision --max-videos 2 --max-images 3
```

**Логика `_worker()`:** каждый вызов выбирает независимые sawtooth-паттерны для изображений и видео, строит сообщение через `build_mixed_vision_message()`, отправляет оба типа медиа в одном запросе.

### Важно
- Если оба `--max-images` и `--max-videos` равны 0 — CLI выдаёт ошибку
- Если ни `--images` ни `--videos` не указаны, медиа генерируются автоматически в `~/.llm-speed-benchmark/tmp/`
- `send_mixed_stats()` — единая функция отправки статистики, обрабатывает все три режима

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

# ASR transcription model (bench_transcription)
TRANSCRIPTION_BASE_URL=http://192.168.45.10:30070/v1
TRANSCRIPTION_API_KEY=any_key
TRANSCRIPTION_MODEL=Qwen3-ASR-0.6B
TRANSCRIPTION_MODEL_TIMEOUT=1200

# TTS model (bench_tts)
TTS_BASE_URL=http://192.168.45.10:30080/v1
TTS_API_KEY=any_key
TTS_MODEL=Qwen3-TTS-12Hz-0.6B-Base
TTS_MODEL_TIMEOUT=600

# Embedding model (bench_emb)
EMB_BASE_URL=http://192.168.45.10:30004/v1
EMB_API_KEY=any_key
EMB_MODEL=/mnt/extendet_data/models_emb/embeddinggemma-300m
EMB_MODEL_TIMEOUT=600
EMB_MAX_MODEL_LEN=2048
```

Загрузка: `utils.py` → `load_dotenv()` из cwd + `~/.llm-speed-benchmark.env`.

**Приоритет конфигурации:** CLI аргументы > `TRANSCRIPTION_*`/`TTS_*`/`EMB_*` из `.env` > базовые `BASE_URL`/`API_KEY`/`MODEL` из `.env`

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

# bench_vision — три режима
bench_vision                                            # 4 воркера, image-only (авто-генерация)
bench_vision -w 8 -d 120                                # 8 воркеров, 120 сек
bench_vision --images ~/workspace/data/images/          # свои изображения
bench_vision --max-images 4                             # макс 4 изображения в запросе
bench_vision -p "Опиши" -p "Что ты видишь?"             # кастомные промпты

bench_vision --videos ~/workspace/data/videos/          # video-only режим
bench_vision --max-videos 4                             # макс 4 видео в запросе
bench_vision --max-videos 4 -w 8 -d 60                  # video-only, 8 воркеров, 60 сек

bench_vision --max-videos 2 --max-images 3              # MIXED режим: и видео, и изобр. в одном запросе
bench_vision --max-videos 3 --max-images 4 -w 8 -d 60   # mixed, 8 воркеров, 60 сек

# bench_audio
bench_audio                                             # 4 воркера, бандл из 10 WAV с речью
bench_audio -w 8 -d 120                                 # 8 воркеров, 120 сек
bench_audio --audio ~/workspace/data/audio/             # свои аудио (.wav, .mp3)
bench_audio --max-audio 1                               # макс 1 аудио в запросе

# bench_transcription — ASR-бенчмарк (Whisper, Qwen-ASR)
bench_transcription                                     # 4 воркера, бандл аудио, .env_asr
bench_transcription -w 2 -d 120                         # 2 воркера, 120 сек
bench_transcription --audio ~/workspace/data/audio/     # свои аудио
bench_transcription --durations 5 10 30 60              # генерация аудио разной длины
bench_transcription --max-duration 30                   # фильтр по максимальной длине
bench_transcription --language ru                       # язык транскрипции
bench_transcription --mode once                         # один проход (не зацикливание)
bench_transcription --model-timeout 1200                # таймаут для долгих аудио

# bench_tts — TTS-бенчмарк (Qwen3-TTS)
bench_tts                                               # 4 воркера, тексты разной длины
bench_tts -w 2 -d 120                                   # 2 воркера, 120 сек
bench_tts --char-counts 10 50 200 500                   # тексты указанной длины (символы)
bench_tts --voice agata                                 # голос TTS
bench_tts --response-format wav                         # формат аудио
bench_tts --mode once                                   # один проход

# bench_emb — Embedding-бенчмарк (embeddinggemma-300m)
bench_emb                                               # 4 воркера, тексты разной длины
bench_emb -w 8 -d 120                                   # 8 воркеров, 120 сек
bench_emb --char-counts 10 50 200 500                   # тексты указанной длины (символы)
bench_emb --batch-sizes 1 5 10                          # batch-запросы
bench_emb --mode once                                   # один проход

 # Общие аргументы
-u, --base-url    Адрес API (OpenAI-compatible)
-k, --api-key     API ключ
-m, --model       Название модели
--max-context     Переопределение MAX_CONTEXT_TOKENS
```

## Kiểm tra
```bash
pytest
```

## Важные замечания

1. **vLLM сервер** должен быть запущен на `BASE_URL` перед запуском
2. **multiprocessing** — каждый воркер создаёт свой OpenAI клиент
3. **Ctrl+C** — корректно останавливает все процессы
4. **src/ layout** — код пакета в `src/llm_speed_benchmark/`, НЕ в корне
5. **Тесты**: `pytest` (моки OpenAI, не требуют сервера)
6. **bench_vision mixed mode**: если оба `--max-images > 0` и `--max-videos > 0` — создаются две медиа-колонки (Imgs+Vid) вместо одной
7. **send_mixed_stats()**: единая функция отправки статистики для всех трёх режимов bench_vision
8. **bench_transcription vs bench_audio**: bench_transcription использует `audio.transcriptions` endpoint (ASR-модели), bench_audio использует `chat.completions` (мультимодальные LLM). bench_transcription измеряет RTF (Real-Time Factor) и скорость обработки аудио по длине
9. **bench_transcription TRANSCRIPTION_***: в `.env` переменные `TRANSCRIPTION_BASE_URL`, `TRANSCRIPTION_API_KEY`, `TRANSCRIPTION_MODEL`, `TRANSCRIPTION_MODEL_TIMEOUT`. Приоритет: CLI > TRANSCRIPTION_* > BASE_URL/API_KEY/MODEL
10. **bench_transcription --durations**: генерирует аудио указанных длительностей зацикливанием бандла, кэширует в `~/.llm-speed-benchmark/tmp/transcription/`