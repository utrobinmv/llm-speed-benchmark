"""
llm-speed-benchmark -- бенчмарк скорости стриминга LLM через vLLM.

Три режима:
  bench_single  -- последовательные вызовы, накопление контекста
  bench_multi   -- N параллельных процессов, Rich Live-таблица
  bench_vision  -- бенчмарк vision-модели (мультимодальная LLM)
"""

__version__ = "0.2.0"
