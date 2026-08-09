# Kronos Synapse — Quickstart Guide

## Что реализовано

✅ Полный цикл обучения decoder-only трансформера с нуля на PyTorch  
✅ Все этапы (1-8) завершены и протестированы  
✅ CPU-only training на i7-1225U с отличной производительностью  

## Архитектура

- **Модель:** GPT-style decoder-only transformer
- **Размер:** Nano конфигурация (~1.9M параметров)
- **Токенизация:** BPE (HuggingFace tokenizers)
- **Датасет:** TinyStories validation (~5.5M токенов)
- **Weight tying:** Да (экономит 50% параметров эмбеддингов)

## Производительность (Реальные замеры)

**Training:**
- Throughput: **~3,300 tokens/sec** (CPU-only)
- 1 эпоха: **~30 минут**
- 10 эпох: **~4.7 часа**
- RAM: ~800MB при batch_size=8

**Inference:**
- KV-cache speedup: ~1.2x
- Поддержка greedy/temperature/top-k/top-p sampling

## Пошаговая инструкция

### 1. Установка зависимостей

```bash
pip install -e .
```

Основные зависимости:
- torch 2.5+ (CPU build)
- tokenizers 0.20+
- typer 0.12+
- rich 13.7+
- pydantic 2.9+

### 2. Скачивание датасета

```bash
python scripts/download_tinystories.py
```

Скачивает TinyStories validation set (~21.5 MB, ~5.5M токенов).

### 3. Обучение токенизатора

```bash
kronos-synapse tokenize data/raw/tinystories.txt \
  --vocab-size 8192 \
  --output data/tokenizer
```

Создаёт BPE токенизатор с vocab_size=8192.  
Время: ~1-2 минуты.

### 4. Подготовка датасета

```bash
python scripts/prepare_dataset.py
```

Токенизирует raw text и сохраняет в `data/processed/train.pt`.  
Размер: ~42 MB (5.5M токенов в формате tensor).

### 5. Проверка конфигурации

```bash
kronos-synapse info --config configs/nano.yaml
```

Показывает параметры модели:
- Layers: 4
- Heads: 4
- d_model: 128
- Context: 256
- Vocab: 8192
- **Total Parameters: ~1.9M**

### 6. Профилирование (опционально)

```bash
python scripts/profile_training.py
```

Запускает 100 шагов обучения для замера реального throughput.  
Результат: ~3,300 tokens/sec.

### 7. Обучение модели

```bash
kronos-synapse train \
  --model-config configs/nano.yaml \
  --training-config configs/training.yaml \
  --data data/processed \
  --checkpoint-dir checkpoints/nano
```

**Параметры обучения (configs/training.yaml):**
- batch_size: 8
- learning_rate: 3e-4
- num_epochs: 10
- warmup_steps: 500
- grad_clip: 1.0

**Ожидаемое время:** ~4.7 часа для 10 эпох.

**Что происходит:**
- Rich прогресс-бары для эпох и шагов
- Логирование loss/perplexity/throughput каждые 10 шагов
- Автоматическое сохранение чекпоинтов каждые 1000 шагов
- Финальный чекпоинт: `checkpoints/nano/final.pt`

**Ожидаемая динамика loss:**
- Начало: ~9.0 (случайная инициализация)
- 1 эпоха: ~5.5-6.0
- 10 эпох: ~3.5-4.0 (perplexity ~40-50)

### 8. Тестирование генерации

```bash
python scripts/test_generation.py
```

Проверяет генерацию с разными стратегиями сэмплинга на **untrained** модели.  
Результат: случайный текст (ожидаемо до обучения).

### 9. Чат с обученной моделью

```bash
kronos-synapse chat checkpoints/nano/final.pt \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --max-length 100
```

**Интерактивный режим:**
- Введите сообщение → модель генерирует ответ
- Команда `clear` — очистить историю
- Команда `quit` — выход

**Streaming вывод:** токены печатаются по мере генерации.

### 10. Оценка perplexity

```bash
kronos-synapse evaluate checkpoints/nano/final.pt \
  --data data/processed \
  --batch-size 8
```

Вычисляет perplexity на валидационном датасете.

## Структура чекпоинтов

Каждый чекпоинт (`.pt` файл) содержит:
- `model_state_dict` — веса модели
- `optimizer_state_dict` — состояние оптимизатора
- `scheduler_state_dict` — состояние scheduler
- `epoch`, `step`, `loss` — метаданные обучения
- `metadata` — model_config, training_config, tokens_processed

Чекпоинты можно загружать для:
- Продолжения обучения (`--resume`)
- Inference/chat
- Evaluation

## Конфигурации

### Nano (текущая)
- Параметры: ~1.9M
- Обучение: ~4.7 часа
- Подходит для: быстрой итерации, proof-of-concept

### Small (для масштабирования)
```bash
kronos-synapse info --config configs/small.yaml
```
- Параметры: ~9.1M
- Обучение: значительно дольше (~20-30 часов)
- Лучшее качество генерации

## Типичные проблемы

### 1. "Tokenizer not found"
```bash
# Убедись, что токенизатор обучен:
kronos-synapse tokenize data/raw/tinystories.txt --vocab-size 8192
```

### 2. "Dataset not found"
```bash
# Подготовь датасет:
python scripts/prepare_dataset.py
```

### 3. Out of memory
- Уменьши `batch_size` в `configs/training.yaml`
- Nano модель требует ~800MB RAM при batch_size=8

### 4. Медленное обучение
- Убедись, что используется CPU build torch (не GPU build на CPU)
- Проверь через профилирование: `python scripts/profile_training.py`

## Следующие шаги

### Эксперименты с гиперпараметрами:
- Увеличить `num_epochs` для лучшей сходимости
- Изменить `learning_rate` / `warmup_steps`
- Попробовать `use_mixed_precision: true` (требует AVX2+)

### Масштабирование:
- Обучить Small модель (`configs/small.yaml`)
- Использовать полный TinyStories датасет (не только validation)
- Увеличить `context_length` для длинного контекста

### Улучшение качества:
- Больше эпох обучения (20-50)
- Larger датасет (WikiText, OpenWebText)
- Fine-tuning на диалогах

## Ограничения

Модель такого масштаба (~2M параметров):
- ✅ Генерирует грамматически простые предложения
- ✅ Учится паттернам из данных
- ✅ Демонстрирует работу трансформеров
- ❌ НЕ "понимает" смысл как GPT-3.5/4
- ❌ Высокая склонность к повторениям
- ❌ Короткий эффективный контекст
- ❌ Ограниченная coherence в длинных текстах

**Это proof-of-concept, не production LLM.**

## Тестирование

```bash
# Все тесты
pytest

# С покрытием
pytest --cov=kronos_synapse

# Type checking
mypy kronos_synapse --strict
```

## Поддержка

Все этапы реализованы и протестированы. Проект готов к:
1. Полному обучению (10+ эпох)
2. Экспериментам с гиперпараметрами
3. Масштабированию на larger модели

---

**Статус:** ✅ Полностью реализован и готов к использованию  
**Дата:** Август 2026  
**Версия:** 0.1.0
