# Kronos Synapse

**Decoder-only трансформер, обученный с нуля на PyTorch** — исследовательский CLI-чатбот без использования сторонних LLM.

## 🎯 Цель проекта

Это образовательный/исследовательский проект, демонстрирующий полный цикл создания языковой модели:
- ✅ Собственная архитектура трансформера (GPT-style)
- ✅ Обучение токенизатора (BPE) с нуля
- ✅ Training loop с нуля на PyTorch
- ✅ CPU-only обучение и инференс (Intel i7-1225U)
- ❌ **НЕ** production-ready решение (модель ~2M параметров vs GPT-3.5 175B)

## 🏗️ Архитектура

### Модель: Decoder-only Transformer
- Multi-head self-attention с causal masking
- Pre-norm LayerNorm
- GELU activation в FFN
- Learned positional embeddings
- **Weight tying** (shared input/output embeddings) — экономия 50% параметров на эмбеддингах
- KV-cache для эффективной генерации

### Конфигурации

#### Nano (текущая, для быстрой итерации)
```yaml
Layers: 4
Heads: 4
d_model: 128
d_ff: 512
Context: 256 tokens
Vocab: 8192
Parameters: ~1.9M
Training time: 6-8 часов на 20M токенов (TinyStories)
```

#### Small (для будущего масштабирования)
```yaml
Layers: 6
Heads: 8
d_model: 256
d_ff: 1024
Context: 512 tokens
Vocab: 16384
Parameters: ~9.1M
Training time: значительно дольше
```

## Быстрый старт

### Установка
```bash
# Python 3.12+ required
pip install -e .

# Для разработки
pip install -e ".[dev]"
```

### 1. Скачать датасет
```bash
python scripts/download_tinystories.py
```

### 2. Обучить токенизатор
```bash
kronos-synapse tokenize data/raw/tinystories.txt --vocab-size 8192 --output data/tokenizer
```

### 3. Подготовить датасет
```bash
python scripts/prepare_dataset.py
```

### 4. Обучить модель
```bash
kronos-synapse train \
  --model-config configs/nano.yaml \
  --training-config configs/training.yaml \
  --data data/processed \
  --checkpoint-dir checkpoints/nano
```

**Ожидаемое время обучения (i7-1225U, CPU-only):**
- 1 эпоха: ~30 минут
- 10 эпох: **~4.7 часа**
- Throughput: **~3,300 tokens/sec**

### 5. Чат с моделью
```bash
kronos-synapse chat checkpoints/nano/final.pt \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9
```

### 6. Оценить качество
```bash
kronos-synapse evaluate checkpoints/nano/final.pt --data data/processed
```

## 📂 Структура проекта

```
kronos_synapse/
├── kronos_synapse/
│   ├── config/          # Pydantic модели конфигурации
│   ├── tokenizer/       # BPE токенизатор (HuggingFace tokenizers)
│   ├── model/           # Архитектура трансформера
│   ├── data/            # Датасет и dataloader
│   ├── training/        # Training loop, optimizer, checkpointing
│   ├── inference/       # Генерация с KV-cache, sampling
│   ├── chat/            # CLI чат-интерфейс
│   └── cli.py           # Typer CLI entrypoint
├── configs/             # YAML конфигурации моделей
├── tests/               # Unit-тесты (pytest)
└── scripts/             # Утилиты (скачивание датасета и т.д.)
```

## ⚙️ Технический стек

- **Framework**: PyTorch 2.5+ (CPU-only)
- **Токенизация**: HuggingFace `tokenizers` (BPE)
- **CLI**: Typer + Rich (прогресс-бары, цветной вывод)
- **Конфигурация**: Pydantic + YAML
- **Типизация**: mypy --strict (100% type hints)
- **Тесты**: pytest

## 🎓 Датасет

**TinyStories** (~500MB, ~20M токенов):
- Простые короткие истории на английском
- Ограниченная лексика и грамматика
- Идеально для быстрой проверки архитектуры на малой модели
- Источник: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

## ⚠️ Известные ограничения

### Что модель МОЖЕТ:
- ✅ Генерировать грамматически простые предложения
- ✅ Поддерживать короткий контекст (2-3 предложения)
- ✅ Имитировать стиль TinyStories
- ✅ Завершать начатые фразы

### Что модель НЕ МОЖЕТ:
- ❌ "Понимать" смысл на уровне GPT-3.5/4
- ❌ Длинные связные диалоги
- ❌ Следование сложным инструкциям
- ❌ Фактическая точность (будет "галлюцинировать")
- ❌ Мультиязычность (обучена только на английском)

**Причина:** Масштаб модели ~2M параметров vs 175B у GPT-3.5. Это демонстрация архитектуры, а не замена LLM.

## 🔬 Производительность (i7-1225U, Реальные замеры)

### Training (профилирование на 100 шагах):
- **Throughput: ~3,300 tokens/sec** (CPU-only, batch_size=8)
- Memory: ~800MB RAM
- **1 эпоха (5.5M токенов): ~30 минут**
- **10 эпох: ~4.7 часа**
- Loss динамика: 8.4 → 5.6 за первые 100 шагов

### Inference (untrained model):
- Генерация: с KV-cache
- KV-cache speedup: ~1.2x на коротких последовательностях
- Поддержка различных стратегий сэмплинга (greedy/temperature/top-k/top-p)

**Примечание:** Все цифры получены профилированием на реальном железе, не теоретические оценки.

## 🧪 Тестирование

```bash
# Все тесты
pytest

# С покрытием
pytest --cov=kronos_synapse --cov-report=html

# Только unit-тесты
pytest tests/test_model.py

# Type checking
mypy kronos_synapse --strict
```

## 📝 Разработка

### Code style
```bash
# Форматирование
black kronos_synapse tests

# Linting
ruff check kronos_synapse tests

# Type checking
mypy kronos_synapse --strict
```

### Добавление новой конфигурации модели
1. Создать `configs/my_config.yaml`
2. Проверить параметры: `kronos-synapse info --config configs/my_config.yaml`
3. Убедиться, что `d_model % n_heads == 0`
4. Оценить время обучения и требования к RAM

## 🛣️ Roadmap

- [x] Этап 1: Инфраструктура (конфигурация, CLI skeleton)
- [x] Этап 2: Токенизация (BPE trainer)
- [x] Этап 3: Архитектура модели
- [x] Этап 4: Датасет и DataLoader
- [x] Этап 5: Training loop + профилирование
- [x] Этап 6: Inference (генерация с KV-cache)
- [x] Этап 7: Chat интерфейс
- [x] Этап 8: Eval + финализация

**Проект полностью реализован!** Готов к обучению на полном датасете.

## 📚 Обучающие материалы

Проект реализует концепции из:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT)

## 📄 Лицензия

MIT

---

**Дисклеймер:** Это исследовательский проект для изучения архитектуры трансформеров. Не используйте модель для production-задач, требующих фактической точности или серьёзного понимания языка.
