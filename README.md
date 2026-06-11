# LLM Speed Benchmark

Бенчмарк скорости стриминга LLM через OpenAI-compatible API (vLLM, Ollama, и любой совместимый сервер).

Два режима: **однопоточный** и **многопроцессный** с Rich Live-таблицей.

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

После установки в PATH появляются команды `bench_single` и `bench_multi`.

## Настройка

Редактируйте `.env` в рабочей директории или создайте `~/.llm-speed-benchmark.env`:

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
bench_single                          # Бежит до лимита контекста
bench_single --duration 60            # Ограничение по времени (60 сек)
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
```

**Особенности:**
- Каждый воркер — независимый процесс с уникальным KV-кэшем
- При достижении лимита контекста — автоматический новый раунд
- Live-обновление метрик каждые 0.5 сек
- Остановка: `Ctrl+C`

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
│   └── utils.py                     # Общие утилиты
├── tests/                           # Тесты (pytest)
│   ├── __init__.py
│   └── test_bench_single.py
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
