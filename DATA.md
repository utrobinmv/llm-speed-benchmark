# DATA.md -- Описание данных проекта

## BEAM dataset

**Источник:** https://huggingface.co/datasets/Mohammadta/BEAM
**Лайцензия:** cc-by-sa-4.0
**Путь:** `~/workspace/data/llm-speed-benchmark/datasets/beam/`

### Скачивание

```bash
hf download --repo-type dataset Mohammadta/BEAM \
    --local-dir ~/workspace/data/llm-speed-benchmark/datasets/beam/
```

### Структура

BEAM содержит multi-turn conversations для оценки long-term memory в LLM.
Данные хранятся в Parquet файлах по сплитам:

| Сплит | Файл | Conversations | Токенов (прибл.) | Размер |
|-------|------|--------------|-------------------|--------|
| 100K | `data/100K-*.parquet` | 20 | 133K -- 300K | ~5 MB |
| 500K | `data/500K-*.parquet` | 35 | 430K -- 1.1M | ~34 MB |
| 1M | `data/1M-*.parquet` | 35 | 947K -- 2.4M | ~66 MB |

### Формат Parquet

Колонки:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `conversation_id` | int | Уникальный ID conversation |
| `chat` | numpy array of numpy arrays | Сообщения диалога. Каждый внутренний массив -- группа сообщений (turn) с dict: `content`, `role` (user/assistant), `id`, `index`, `question_type`, `time_anchor` |
| `conversation_seed` | dict | category, id, subtopics, theme, title |
| `narratives` | str | Описание narrative |
| `user_profile` | dict | user_info, user_relationships |
| `conversation_plan` | str | План conversation |
| `user_questions` | list | Вопросы пользователя |
| `probing_questions` | str | Вопросы для оценки memory |

### Использование в бенчмарке

Для бенчмарка long context воркеры загружают conversation из выбранного сплита,
преобразуют в формат OpenAI API messages и передают как initial context.
Далее воркер продолжает генерацию с новым промптом, измеряя скорость стриминга
при длинном контексте.

Пример выбора conversation:

```python
from llm_speed_benchmark.long_context import LongContextDataset

ds = LongContextDataset(
    data_dir=os.path.expanduser("~/workspace/data/llm-speed-benchmark/datasets"),
)
ds.load(split="100K")

# Получить messages с обрезкой до 128K токенов
messages = ds.get_messages(conversation_id=0, max_tokens=128_000)
```
