# 🔍 Валидация анонимизаторов - validate_simple.py

## 📋 Назначение

Скрипт `scripts/validate_simple.py` предназначен для быстрой валидации и сравнения производительности различных анонимизаторов без сохранения детальных отчетов. Основной инструмент для тестирования качества распознавания персональных данных.

## 🛠️ Основные функции

- Тестирование всех доступных анонимизаторов на одном датасете
- Расчет метрик точности, полноты и F1-score
- Умное сопоставление сущностей с учетом перекрытий
- Автоматическое объединение соседних PERSON сущностей
- Поддержка различных форматов входных данных

## 📥 Входные параметры

### Командная строка:
```bash
# Базовая валидация с автоопределением датасета
python scripts/validate_simple.py

# Валидация с указанием конкретного датасета
python scripts/validate_simple.py --dataset data/generated/dataset.json

# Валидация с настройкой порога перекрытия
python scripts/validate_simple.py --overlap-threshold 0.3

# Валидация только выбранных анонимизаторов
python scripts/validate_simple.py --models natasha,spacy
```

### Программный интерфейс:
```python
from scripts.validate_simple import validate_anonymizer, load_test_dataset

# Загрузка тестового датасета
test_data = load_test_dataset("data/test_dataset.json")

# Валидация конкретного анонимизатора
from anonymizers.natasha_enhanced.enhanced_natasha import EnhancedNatashaAnonymizer
anonymizer = EnhancedNatashaAnonymizer()
results = validate_anonymizer(anonymizer, "Natasha Enhanced", test_data)

# Просмотр результатов
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1-Score: {results['f1_score']:.3f}")
```

## 🎯 Поддерживаемые анонимизаторы

- **NatashaEnhanced** - Анонимизатор на основе библиотеки Natasha
- **SpacyExtended** - Расширенный анонимизатор на spaCy
- **BertTransformers** - BERT-based анонимизатор
- **HybridAdvanced** - Гибридный анонимизатор
- **RegExpBaseline** - Базовый анонимизатор на регулярных выражениях

## 📤 Выходные данные

### Консольный отчет:
```
🔍 Простая валидация анонимизаторов
========================================
📂 Загрузка датасета: data/generated/dataset.json
✅ Обнаружен стандартный формат. Загрузка...
👍 Загрузка завершена. 100 диалогов готово к валидации.

🧠 Валидация: Natasha Enhanced
⏱️  Время обработки: 2.34 сек
📊 Результаты:
   Precision: 0.856
   Recall: 0.742
   F1-Score: 0.795
   Обработано сущностей: 145/167

📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:
========================================
🥇 1. Natasha Enhanced    - F1: 0.795
🥈 2. Spacy Extended      - F1: 0.763
🥉 3. Hybrid Advanced     - F1: 0.741
```

### Результаты валидации:
```python
results = {
    'precision': 0.856,
    'recall': 0.742,
    'f1_score': 0.795,
    'true_positives': 124,
    'false_positives': 21,
    'false_negatives': 43,
    'processing_time': 2.34,
    'total_entities_found': 145,
    'total_entities_expected': 167
}
```

## 🔧 Внутренние компоненты

### Алгоритм умного сопоставления:
```python
def smart_entity_matching(true_entities, predicted_entities, overlap_threshold=0.2):
    """
    Сопоставление с учетом:
    - Точных совпадений
    - Частичных перекрытий
    - Объединения соседних PERSON сущностей
    """
```

### Функции улучшения:
- `merge_person_entities()` - Объединение соседних имен
- `improve_extracted_entities()` - Постобработка результатов
- `calculate_overlap_score()` - Расчет степени перекрытия

### Поддерживаемые форматы данных:
```python
# Стандартный формат с metadata
{
  "metadata": {...},
  "dialogs": [
    {
      "id": 1,
      "text": "Текст диалога",
      "entities": [
        {
          "start": 10,
          "end": 20,
          "type": "PERSON",
          "text": "Иван Петров"
        }
      ]
    }
  ]
}

# Упрощенный формат
[
  {
    "text": "Текст диалога",
    "personal_data": [...]
  }
]
```

## 📊 Метрики качества

### Основные метрики:
- **Precision** - Доля правильно найденных сущностей
- **Recall** - Доля найденных от всех существующих
- **F1-Score** - Гармоническое среднее precision и recall

### Настройки сопоставления:
- **overlap_threshold** - Минимальное перекрытие для засчета (по умолчанию 0.2)
- **merge_persons** - Объединение соседних имен (включено)
- **smart_matching** - Умное сопоставление с частичными совпадениями

## 📦 Зависимости

```txt
json
time
pathlib
typing
argparse
anonymizers.*  # Все анонимизаторы проекта
```

## ⚠️ Примечания

- **Автопоиск датасета**: Автоматически ищет доступные тестовые файлы
- **Умное сопоставление**: Учитывает частичные перекрытия и варианты написания
- **Быстрые проверки**: Оптимизирован для частого использования
- **Обработка ошибок**: Продолжает работу даже при недоступности некоторых анонимизаторов
- **Без сохранения**: Результаты только в консоли для быстрого анализа

## 🔗 Связанные файлы

- `scripts/validate_deeppavlov_improved.py` - Специализированная валидация DeepPavlov
- `anonymizers/` - Модули анонимизаторов
- `data/test_dataset.json` - Тестовые данные
- `docs/reference/BENCHMARKS.md` - Подробные результаты тестирования 