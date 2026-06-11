"""
llm-speed-benchmark — бенчмарк скорости стриминга LLM через vLLM.

Два режима:
  bench_single  — последовательные вызовы, накопление контекста
  bench_multi   — N параллельных процессов, Rich Live-таблица
"""

__version__ = "0.2.0"
