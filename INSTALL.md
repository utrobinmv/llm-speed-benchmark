# INSTALL.md — Установка llm-speed-benchmark

## Быстрая установка (из Git)

```bash
pip install git+https://github.com/utrobinmv/llm-speed-benchmark.git
```

После установки доступны команды `bench_single` и `bench_multi`.

## Установка из локальной копии

```bash
git clone https://github.com/utrobinmv/llm-speed-benchmark.git
cd llm-speed-benchmark
pip install -e .
```

Флаг `-e` (editable mode) — изменения в коде сразу отражаются без переустановки.

## Установка с dev-зависимостями

```bash
pip install -e ".[dev]"
```

Добавляет: `ruff`, `mypy`, `pytest`.

## Развёртывание venv с нуля

```bash
python3 -m venv ~/workspace/venvs/llm-speed-benchmark/venv
source ~/workspace/venvs/llm-speed-benchmark/venv/bin/activate
cd ~/workspace/projects/llm-speed-benchmark
pip install --upgrade pip
pip install -e ".[dev]"
```

## Проверка установки

```bash
bench_single --help
bench_multi --help
```

## Настройка (.env)

Создайте `.env` в рабочей директории или `~/.llm-speed-benchmark.env`:

```ini
BASE_URL=http://localhost:8000/v1
API_KEY=sk-vllm-qwen3.5-0.8b
MODEL=qwen3.5-0.8b
MAX_CONTEXT_TOKENS=262144
```

## Тесты

```bash
source ~/workspace/venvs/llm-speed-benchmark/venv/bin/activate
cd ~/workspace/projects/llm-speed-benchmark
pytest
```

## Сборка дистрибутива

```bash
pip install build
python -m build
# Результат: dist/llm_speed_benchmark-0.2.0-py3-none-any.whl
```
