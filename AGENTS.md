# AGENTS.md — Инструкции для AI-агента

## Контекст проекта

**llm-speed-benchmark** — pip-устанавливаемый пакет для бенчмарка скорости стриминга LLM через OpenAI-compatible API (vLLM, Ollama и др.).

Два режима:
- `bench_single` — последовательные вызовы, накопление контекста, статистика
- `bench_multi` — N параллельных процессов (multiprocessing), Rich Live-таблица

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
│   └── utils.py                     # get_client(), truncate_history(), progress_bar(), format_time()
├── tests/
│   ├── __init__.py
│   └── test_bench_single.py         # pytest (mock OpenAI)
├── .venv                            # source .venv → активация
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
```

## Зависимости

| Пакет | Зачем |
|---|---|
| `openai>=1.0.0` | OpenAI-compatible HTTP клиент |
| `python-dotenv>=1.0.0` | Загрузка `.env` |
| `rich>=13.0.0` | Rich Live-таблица в `bench_multi` |

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

```bash
bench_single                          # До лимита контекста
bench_single --duration 60            # 60 секунд

bench_multi                           # 4 воркера, до Ctrl+C
bench_multi -w 8 -d 60                # 8 воркеров, 60 сек
bench_multi --response-width 120      # Широкая колонка
bench_multi --max-context 32768       # Переопределение лимита
```

## Важные замечания

1. **vLLM сервер** должен быть запущен на `BASE_URL` перед запуском
2. **multiprocessing** — каждый воркер создаёт свой OpenAI клиент
3. **Ctrl+C** — корректно останавливает все процессы
4. **src/ layout** — код пакета в `src/llm_speed_benchmark/`, НЕ в корне
5. **Тесты**: `pytest` (моки OpenAI, не требуют сервера)
