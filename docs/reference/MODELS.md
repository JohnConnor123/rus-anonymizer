# 🤖 Описание моделей и анонимизаторов

## 📋 Обзор анонимизаторов

Система **Anonymizer** включает 6 различных анонимизаторов, каждый из которых имеет свои преимущества и области применения.

| Анонимизатор | Технология | F1-Score | Скорость | Рекомендуемое использование |
|-------------|------------|----------|----------|---------------------------|
| **Hybrid Advanced** | Комбинированный | 0.85 | Средняя | 🏆 Лучший общий выбор |
| **Natasha Enhanced** | Морфологический анализ | 0.75 | Быстрая | Для русских имен/мест |
| **SpaCy Extended** | NLP пайплайн | 0.68 | Быстрая | Универсальный |
| **DeepPavlov NER** | Transformer (BERT) | 0.47 | Медленная | Для сложных документов |
| **RegExp Baseline** | Регулярные выражения | 0.45 | Очень быстрая | Для простых паттернов |
| **Transformers BERT** | BERT + fine-tuning | Варьируется | Медленная | Экспериментальный |

## 🔧 Детальное описание анонимизаторов

### 1. 🏆 Hybrid Advanced Anonymizer

**Файл:** `anonymizers/hybrid_advanced/hybrid_anonymizer.py`

**Описание:** Комбинирует несколько подходов для достижения максимального качества анонимизации.

**Технология:**
- Объединяет результаты Natasha, SpaCy и RegExp
- Использует умное слияние результатов с весами
- Применяет постобработку для устранения конфликтов

**Поддерживаемые типы сущностей:**
```python
SUPPORTED_ENTITIES = [
    'PERSON',      # ФИО
    'PHONE',       # Телефоны  
    'EMAIL',       # Email адреса
    'LOCATION',    # Адреса, города
    'ORGANIZATION',# Организации
    'AGE',         # Возраст
    'DATE',        # Даты рождения
    'DOCUMENT',    # Номера документов
    'BANK_CARD',   # Банковские карты
    'IP_ADDRESS'   # IP адреса
]
```

**Преимущества:**
- ✅ Высокое качество (F1: 0.85)
- ✅ Обрабатывает сложные случаи
- ✅ Устойчив к опечаткам
- ✅ Минимум ложных срабатываний

**Недостатки:**
- ❌ Медленнее других методов
- ❌ Требует больше памяти

**Пример использования:**
```python
from anonymizers.hybrid_advanced.hybrid_anonymizer import HybridAnonymizer

anonymizer = HybridAnonymizer(
    aggressiveness=0.8,
    enable_postprocessing=True,
    weights={'natasha': 0.4, 'spacy': 0.3, 'regexp': 0.3}
)

text = "Звоните Петрову Ивану Сергеевичу по номеру +7-926-123-45-67"
result = anonymizer.anonymize(text)
print(result)
# "Звоните <PERSON> по номеру <PHONE>"
```

### 2. 🌟 Natasha Enhanced Anonymizer

**Файл:** `anonymizers/natasha_enhanced/natasha_anonymizer.py`

**Описание:** Основан на библиотеке Natasha для морфологического анализа русского языка.

**Технология:**
- Использует Natasha + Yargy для парсинга
- Морфологический анализ с pymorphy2
- Специализирован для русского языка

**Особенности:**
```python
NATASHA_ENTITIES = {
    'PER': 'PERSON',        # Персоны
    'LOC': 'LOCATION',      # Локации  
    'ORG': 'ORGANIZATION',  # Организации
}

CUSTOM_PATTERNS = {
    'PHONE': r'(\+7|8)[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}',
    'EMAIL': r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    'AGE': r'\b(?:мне|возраст|лет)\s+(\d{1,2})\b'
}
```

**Преимущества:**
- ✅ Отличная работа с русскими именами
- ✅ Высокая скорость обработки  
- ✅ Хорошее качество (F1: 0.75)
- ✅ Низкое потребление памяти

**Недостатки:**
- ❌ Ориентирован только на русский язык
- ❌ Ограниченная работа с документами

**Пример использования:**
```python
from anonymizers.natasha_enhanced.natasha_anonymizer import NatashaAnonymizer

anonymizer = NatashaAnonymizer(
    aggressiveness=0.7,
    use_morph_analysis=True,
    custom_replacements={'PERSON': '[ПЕРСОНА]'}
)
```

### 3. 🌍 SpaCy Extended Anonymizer

**Файл:** `anonymizers/spacy_extended/spacy_anonymizer.py`

**Описание:** Использует SpaCy с русской моделью `ru_core_news_sm`.

**Технология:**
- SpaCy NLP пайплайн для русского языка
- Именованное распознавание сущностей (NER)
- Дополнительные регулярные выражения

**Модель:** `ru_core_news_sm`
```python
SPACY_TO_CUSTOM = {
    'PER': 'PERSON',
    'LOC': 'LOCATION', 
    'ORG': 'ORGANIZATION',
    'MISC': 'OTHER'
}
```

**Преимущества:**
- ✅ Быстрая обработка
- ✅ Хорошая универсальность
- ✅ Стабильная работа
- ✅ Простота использования

**Недостатки:**
- ❌ Требует загрузки модели
- ❌ Иногда пропускает специфичные паттерны

**Пример использования:**
```python
from anonymizers.spacy_extended.spacy_anonymizer import SpacyAnonymizer

anonymizer = SpacyAnonymizer(
    model_name='ru_core_news_sm',
    confidence_threshold=0.7
)
```

### 4. 🧠 DeepPavlov NER Anonymizer

**Файл:** `anonymizers/deeppavlov_ner/deeppavlov_anonymizer.py`

**Описание:** Использует DeepPavlov BERT для распознавания именованных сущностей.

**Технология:**
- Предобученная BERT модель для русского языка
- Модель: `ner_rus_bert`
- Глубокое обучение для NER

**Конфигурация:**
```python
DEEPPAVLOV_CONFIG = {
    'model': 'ner_rus_bert',
    'batch_size': 32,
    'max_seq_length': 512
}

ENTITY_MAPPING = {
    'B-PER': 'PERSON', 'I-PER': 'PERSON',
    'B-LOC': 'LOCATION', 'I-LOC': 'LOCATION',
    'B-ORG': 'ORGANIZATION', 'I-ORG': 'ORGANIZATION'
}
```

**Преимущества:**
- ✅ Понимает контекст
- ✅ Хорошо работает со сложными текстами
- ✅ Высокая точность для известных сущностей

**Недостатки:**
- ❌ Очень медленная работа
- ❌ Требует много памяти (GPU рекомендуется)
- ❌ Сложная установка зависимостей

**Пример использования:**
```python
from anonymizers.deeppavlov_ner.deeppavlov_anonymizer import DeepPavlovAnonymizer

# Требует отдельного окружения venv-deeppavlov
anonymizer = DeepPavlovAnonymizer(
    model_config='ner_rus_bert',
    batch_size=16
)
```

### 5. ⚡ RegExp Baseline Anonymizer

**Файл:** `anonymizers/regexp_baseline/regexp_anonymizer.py`

**Описание:** Базовый анонимизатор на регулярных выражениях.

**Технология:**
- Только регулярные выражения
- Заранее определенные паттерны
- Высокая скорость обработки

**Паттерны:**
```python
PATTERNS = {
    'PERSON': [
        r'\b[А-ЯЁ][а-яё]{2,15}\s+[А-ЯЁ][а-яё]{2,20}\b',  # Имя Фамилия
        r'\b[А-ЯЁ][а-яё]{2,15}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b'   # Фамилия И.О.
    ],
    'PHONE': [
        r'\+7[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}',
        r'8[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}'
    ],
    'EMAIL': [
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    ],
    'BANK_CARD': [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    ]
}
```

**Преимущества:**
- ✅ Максимальная скорость
- ✅ Высокая точность для известных паттернов
- ✅ Простота и надежность
- ✅ Низкое потребление ресурсов

**Недостатки:**
- ❌ Низкая полнота (recall)
- ❌ Не понимает контекст
- ❌ Много ложных пропусков

**Пример использования:**
```python
from anonymizers.regexp_baseline.regexp_anonymizer import RegExpBaselineAnonymizer

anonymizer = RegExpBaselineAnonymizer(
    custom_patterns={
        'CUSTOM_ID': r'\b[A-Z]{2}\d{6}\b'
    }
)
```

### 6. 🔬 Transformers BERT Anonymizer

**Файл:** `anonymizers/transformers_bert/transformers_anonymizer.py`

**Описание:** Экспериментальный анонимизатор на базе Transformers.

**Технология:**
- Hugging Face Transformers
- Fine-tuned BERT для русского NER
- Экспериментальные подходы

**Модели:**
```python
AVAILABLE_MODELS = [
    'DeepPavlov/rubert-base-cased-sentence',
    'sberbank-ai/ruBert-base', 
    'cointegrated/rubert-tiny2'
]
```

**Особенности:**
- 🧪 Экспериментальный статус
- 🔄 Активная разработка
- ⚙️ Настраиваемые модели

## 🎯 Рекомендации по выбору

### По задачам:

**Для продакшена (высокое качество):**
```python
# Лучший выбор - гибридный
from anonymizers.hybrid_advanced.hybrid_anonymizer import HybridAnonymizer
anonymizer = HybridAnonymizer(aggressiveness=0.8)
```

**Для быстрой обработки:**
```python
# Natasha для баланса скорости и качества
from anonymizers.natasha_enhanced.natasha_anonymizer import NatashaAnonymizer
anonymizer = NatashaAnonymizer(aggressiveness=0.7)
```

**Для максимальной скорости:**
```python
# RegExp для простых случаев
from anonymizers.regexp_baseline.regexp_anonymizer import RegExpBaselineAnonymizer
anonymizer = RegExpBaselineAnonymizer()
```

### По типам данных:

- **Персональные письма:** `NatashaEnhanced` или `HybridAdvanced`
- **Официальные документы:** `DeepPavlov` или `HybridAdvanced`
- **Логи/технические тексты:** `RegExpBaseline` или `SpacyExtended`
- **Смешанный контент:** `HybridAdvanced`

## 📊 Сравнительные метрики

### Производительность (тексты ~1000 символов):

| Анонимизатор | Время (мс) | Память (MB) | CPU нагрузка |
|-------------|------------|-------------|--------------|
| RegExp Baseline | 5-10 | 10-20 | Низкая |
| Natasha Enhanced | 50-100 | 50-100 | Средняя |
| SpaCy Extended | 100-200 | 100-200 | Средняя |
| Hybrid Advanced | 200-400 | 200-300 | Высокая |
| DeepPavlov NER | 1000-3000 | 1000-2000 | Очень высокая |
| Transformers BERT | 500-1500 | 500-1000 | Высокая |

### Качество по типам сущностей:

| Анонимизатор | PERSON | PHONE | EMAIL | LOCATION | ORGANIZATION |
|-------------|--------|-------|-------|----------|--------------|
| Hybrid Advanced | 0.95 | 0.90 | 0.98 | 0.85 | 0.75 |
| Natasha Enhanced | 0.90 | 0.85 | 0.95 | 0.80 | 0.70 |
| SpaCy Extended | 0.75 | 0.70 | 0.90 | 0.75 | 0.65 |
| DeepPavlov NER | 0.80 | 0.40 | 0.60 | 0.70 | 0.80 |
| RegExp Baseline | 0.60 | 0.95 | 0.98 | 0.30 | 0.20 |

## 🔧 Настройка и оптимизация

### Общие параметры:

```python
# Базовые настройки для всех анонимизаторов
base_config = {
    'aggressiveness': 0.7,           # Агрессивность 0.0-1.0
    'custom_replacements': {},       # Пользовательские замены
    'excluded_entities': [],         # Исключаемые типы
    'confidence_threshold': 0.5      # Порог уверенности
}
```

### Специфичные настройки:

```python
# Для Hybrid Advanced
hybrid_config = {
    'weights': {'natasha': 0.4, 'spacy': 0.3, 'regexp': 0.3},
    'enable_postprocessing': True,
    'resolve_conflicts': True
}

# Для DeepPavlov  
deeppavlov_config = {
    'batch_size': 32,
    'max_seq_length': 512,
    'use_gpu': False
}
```

## 🚀 Будущие улучшения

### Планируемые модели:
- **GPT-based Anonymizer** - использование больших языковых моделей
- **LSTM-CRF Anonymizer** - классический подход к NER
- **Multi-language Support** - поддержка других языков

### Улучшения существующих:
- Оптимизация скорости Hybrid Advanced
- Расширение паттернов RegExp Baseline  
- Интеграция новых моделей в Transformers BERT
