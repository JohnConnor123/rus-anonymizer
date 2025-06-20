# 🔄 Обработка данных - data_processor.py

## 📋 Назначение

Скрипт `scripts/data_processor.py` предназначен для обработки и преобразования датасетов диалогов. Выполняет нормализацию данных, объединение датасетов, фильтрацию и подготовку данных для обучения моделей анонимизации.

## 🛠️ Основные функции

- Нормализация форматов датасетов из разных источников
- Объединение нескольких датасетов в один
- Фильтрация диалогов по критериям качества
- Статистический анализ содержимого
- Удаление дубликатов и некорректных записей
- Преобразование форматов аннотации

## 📥 Входные параметры

### Командная строка:
```bash
# Базовая обработка датасета
python scripts/data_processor.py --input data/raw/dataset.json --output data/processed/clean_dataset.json

# Объединение нескольких файлов
python scripts/data_processor.py --merge data/generated/*.json --output data/processed/merged.json

# Фильтрация по качеству
python scripts/data_processor.py --input dataset.json --filter-quality 0.8 --output clean.json

# Анализ без сохранения
python scripts/data_processor.py --input dataset.json --analyze-only
```

### Программный интерфейс:
```python
from scripts.data_processor import DataProcessor

processor = DataProcessor()

# Загрузка и обработка датасета
dataset = processor.load_dataset("data/raw/dataset.json")
processed = processor.process_dataset(dataset, quality_threshold=0.8)
processor.save_dataset(processed, "data/processed/clean.json")

# Объединение датасетов
merged = processor.merge_datasets([
    "dataset1.json", 
    "dataset2.json", 
    "dataset3.json"
])
```

## 🎯 Операции обработки

### 1. Нормализация данных:
- Унификация полей диалогов
- Стандартизация типов сущностей
- Исправление кодировки текста
- Нормализация позиций сущностей

### 2. Фильтрация качества:
```python
QUALITY_CRITERIA = {
    'min_text_length': 10,        # Минимальная длина текста
    'max_text_length': 5000,      # Максимальная длина текста
    'min_entities': 0,            # Минимум сущностей
    'max_entities_ratio': 0.5,    # Максимум доли сущностей
    'valid_encoding': True,       # Корректная кодировка
    'no_duplicates': True         # Отсутствие дубликатов
}
```

### 3. Объединение датасетов:
- Слияние метаданных
- Устранение конфликтов ID
- Объединение статистики
- Проверка совместимости форматов

### 4. Анализ содержимого:
- Распределение длин текстов
- Частота типов сущностей  
- Покрытие персональных данных
- Качество аннотации

## 📤 Выходные данные

### Обработанный датасет:
```json
{
  "metadata": {
    "processed_at": "2024-01-20T15:30:00",
    "source_files": ["dataset1.json", "dataset2.json"],
    "total_dialogs": 150,
    "quality_threshold": 0.8,
    "processing_stats": {
      "removed_duplicates": 5,
      "filtered_low_quality": 10,
      "normalized_entities": 45
    }
  },
  "dialogs": [...]
}
```

### Отчет о обработке:
```
📊 Отчет обработки данных:
==================================================
📁 Входные файлы: 3
📋 Исходных диалогов: 200
✅ Обработано успешно: 180 (90.0%)
🗑️ Удалено дубликатов: 15
❌ Отфильтровано низкого качества: 5

📈 Статистика по сущностям:
   PERSON: 120 → 115 (-4.2%)
   PHONE: 80 → 78 (-2.5%)
   EMAIL: 45 → 44 (-2.2%)

📊 Распределение по длине текста:
   Короткие (< 100 символов): 20 (11.1%)
   Средние (100-500): 130 (72.2%)
   Длинные (> 500): 30 (16.7%)
```

## 🔧 Внутренние компоненты

### Класс DataProcessor:
```python
class DataProcessor:
    def __init__(self, config: Dict = None):
        self.quality_criteria = config or DEFAULT_CRITERIA
    
    def load_dataset(self, filepath: str) -> Dict
    def process_dataset(self, dataset: Dict, **kwargs) -> Dict
    def merge_datasets(self, filepaths: List[str]) -> Dict
    def analyze_dataset(self, dataset: Dict) -> Dict
    def save_dataset(self, dataset: Dict, filepath: str)
```

### Методы валидации:
- `_validate_dialog()` - Проверка корректности диалога
- `_normalize_entities()` - Нормализация сущностей
- `_detect_duplicates()` - Поиск дубликатов
- `_calculate_quality_score()` - Расчет метрики качества

### Конфигурация обработки:
```python
PROCESSING_CONFIG = {
    'normalize_unicode': True,
    'fix_entity_positions': True,
    'remove_empty_dialogs': True,
    'merge_overlapping_entities': True,
    'standardize_entity_types': True
}
```

## 📦 Зависимости

```txt
json
re
unicodedata
collections
typing
pathlib
datetime
```

## ⚠️ Примечания

- **Бережная обработка**: Сохраняет исходные данные при возможности
- **Конфигурируемость**: Настраиваемые критерии качества
- **Производительность**: Эффективная обработка больших файлов
- **Логирование**: Подробные отчеты о всех изменениях
- **Обратимость**: Возможность восстановления исходных данных

## 🔗 Связанные файлы

- `scripts/merge_datasets.py` - Специализированное слияние
- `scripts/split_dataset.py` - Разделение датасетов
- `scripts/metadata_utils.py` - Утилиты работы с метаданными
- `tests/validate_simple.py` - Проверка качества после обработки 