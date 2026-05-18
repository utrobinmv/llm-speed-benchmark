# LLM Speed Benchmark

Бенчмарк скорости стриминга LLM (vLLM) — однопоточный и многопроцессный режимы.

## Установка

```bash
pip install -e .
```

После установки в PATH появятся команды `bench_single` и `bench_multi`.

## Настройка

Редактируйте `.env`:

```ini
BASE_URL=http://localhost:8000/v1
API_KEY=sk-vllm-qwen3.5-0.8b
MODEL=qwen3.5-0.8b
MAX_CONTEXT_TOKENS=262144
```

- `BASE_URL` — адрес vLLM сервера
- `API_KEY` — ключ (обычно любой, vLLM не проверяет)
- `MODEL` — название модели
- `MAX_CONTEXT_TOKENS` — лимит контекста (85% триггер обрезки)

## Использование

### Одиночный воркер

Без параметров — бежит до лимита контекста:

```bash
bench_single
```

С ограничением по времени:

```bash
bench_single --duration 60
```

### Многопроцессный

```bash
# 4 воркера, 30 секунд
bench_multi --workers 4 --duration 30

# Ширина колонки Response (по умолч. 60)
bench_multi --workers 4 --duration 30 --response-width 120

# Все параметры
bench_multi --workers 4 --duration 60 --response-width 120
```

## Режимы

- **bench_single** — последовательные вызовы, накопление контекста, вывод статистики
- **bench_multi** — N параллельных процессов, Rich Live таблица, live-обновление ответа

Остановка: `Ctrl+C`

## Структура

```
.
├── pyproject.toml       # настройки пакета
├── bench_single.py      # однопоточный бенчмарк
├── bench_multi.py       # многопроцессный бенчмарк
├── .env                 # конфигурация
└── README.md
```
