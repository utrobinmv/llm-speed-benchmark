# AGENTS.md — Инструкции для AI-агента

## Контекст проекта

**llm-speed-benchmark** — бенчмарк скорости стриминга LLM через vLLM. Два режима:
- `bench_single.py` — последовательные вызовы, накопление контекста, статистика
- `bench_multi.py` — N параллельных процессов (multiprocessing), Rich Live-таблица

Проект обёрнут вокруг OpenAI-compatible API (работает с vLLM, Ollama, и любым сервером с OpenAI API).

### Конвенции

- **Python 3.11+** (фактически 3.12, `requires-python = ">=3.10"`)
- **Виртуальное окружение**: `~/venvs/llm-speed-benchmark/venv/`
- **Активация**: `source ~/venvs/llm-speed-benchmark/venv/bin/activate` или `source .venv`
- **Сообщения с пользователем** — на **русском языке**

## Структура проекта

```
llm-speed-benchmark/
├── bench_single.py              # Одиночный воркер (последовательный)
├── bench_multi.py               # Многопроцессный (multiprocessing + Rich Live)
├── bench_multi_requirements.md  # ТЗ/спецификация bench_multi.py
├── pyproject.toml               # Пакетная конфигурация (setuptools)
├── requirements.txt             # Зависимости (pip)
├── .env                         # Конфигурация (BASE_URL, API_KEY, MODEL)
├── .venv                        # Скрипт активации venv
├── .gitignore
├── README.md
└── AGENTS.md                    # ← этот файл
```

## Зависимости

| Пакет | Зачем |
|-------|-------|
| `openai>=1.0.0` | OpenAI-compatible HTTP клиент |
| `python-dotenv>=1.0.0` | Загрузка `.env` |
| `rich>=13.0.0` | Rich Live-таблица в `bench_multi` |

**Установка:**
```bash
source ~/venvs/llm-speed-benchmark/venv/bin/activate
cd ~/projects/llm-speed-benchmark
pip install -e .
```

После установки доступны команды `bench_single` и `bench_multi` через PATH.

## Конфигурация (.env)

```ini
BASE_URL=http://localhost:8000/v1    # Адрес vLLM сервера (OpenAI-compatible API)
API_KEY=sk-vllm-qwen3.5-0.8b         # Ключ (vLLM не проверяет)
MODEL=qwen3.5-0.8b                   # Название модели
MAX_CONTEXT_TOKENS=262144            # Лимит контекста (85% = триггер обрезки)
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

## bench_single.py

Последовательный бенчмарк: делает запросы один за другим, накапливает контекст, при достижении лимита останавливается.

```bash
bench_single                      # Бежит до лимита контекста
bench_single --duration 60        # Ограничение по времени (60 сек)
```

**Логика:**
1. Первый запрос: «Что ты умеешь? Расскажи обо всём максимально подробно.»
2. Последующие: «продолжай»
3. При достижении 85% контекста — обрезает историю (оставляет system)
4. При достижении 100% — останавливается
5. Выводит статистику: prompt tokens, gen tokens, скорость

## bench_multi.py

Многопроцессный бенчмарк: запускает N независимых процессов, каждый с уникальным ID.

```bash
bench_multi                      # 4 воркера, 30 секунд (по умолчанию)
bench_multi --workers 8          # 8 воркеров
bench_multi --workers 4 --duration 60   # 4 воркера, 60 секунд
bench_multi --workers 4 --duration 60 --response-width 120
```

**Архитектура:**
```
┌──────────────────────────────────────────────────┐
│                 Main Process                      │
│  display_loop() → reads Queue → Rich Live table  │
└──────┬───────────────────────────────────────────┘
       │ multiprocessing.Queue
       │
┌──────┼──────┬───────┬──────┐
▼      ▼      ▼       ▼      ▼
Worker 0  Worker 1  ...  Worker N-1
```

**Особенности:**
- Каждый воркер — отдельный процесс (`multiprocessing.Process`)
- Уникальные промпты для разного KV-кэша
- При достижении лимита — **новый раунд** (не остановка)
- Rich Live-таблица обновляется ~1 раз в секунду
- Стоп: Ctrl+C

## .gitignore

```
*.pyc
__pycache__/
*.egg-info
```

## Важные замечания

1. **vLLM сервер** должен быть запущен на `BASE_URL` перед запуском бенчмарка
2. **multiprocessing** — каждый воркер создаёт собственный OpenAI клиент и сессию
3. **Ctrl+C** — корректно останавливает все процессы
4. **MAX_CONTEXT_TOKENS** — лимит контекста модели (262144 по умолчанию = 256K)
