# 🔄 Обработка данных - data_processor.py

## 📋 Назначение

Скрипт `scripts/data_processor.py` предназначен для обработки и преобразования датасетов диалогов. Выполняет объединение датасетов, статистический анализ и подготовку данных для обучения моделей анонимизации.

## 🛠️ Основные функции

- Объединение нескольких датасетов в один
- Статистический анализ содержимого датасетов
- Разделение объединенного датасета на train/val
- Вычисление метрик качества аннотации
- Анализ распределения типов сущностей

## 📥 Входные параметры

### Командная строка:
```bash
# Анализ одного датасета
python scripts/data_processor.py --input data/generated/dataset.json --analyze-only

# Объединение нескольких файлов
python scripts/data_processor.py --merge data/generated/*.json --output data/processed/merged.json

# Объединение с паттерном в кавычках
python scripts/data_processor.py --merge "data/generated/llama*.json" --output data/processed/filtered.json

# Объединение конкретных файлов
python scripts/data_processor.py --merge file1.json file2.json --output combined.json
```

### Программный интерфейс:
```python
from scripts.data_processor import DatasetProcessor

# Создание процессора с файлами для объединения
processor = DatasetProcessor(
    input_files=["dataset1.json", "dataset2.json"],
    output_dir="data/processed"
)

# Объединение датасетов (автоматически загружает все input_files)
merged_data = processor.merge_datasets()

# Анализ объединенного датасета
stats = processor.calculate_stats(merged_data.get('dialogs', []))
print(f"Всего диалогов: {stats.total_dialogs}")
print(f"Диалогов с ПД: {stats.pd_dialogs} ({stats.pd_percentage:.1f}%)")

# Разделение на train/val
train_data, val_data = processor.split_dataset(merged_data, train_ratio=0.7)

# Сохранение результатов
processor.save_dataset(merged_data, "merged_dataset.json")
processor.save_dataset(train_data, "train_dataset.json")
processor.save_dataset(val_data, "val_dataset.json")

# Загрузка ранее сохраненного файла
saved_data = processor.load_dataset("data/processed/merged_dataset.json")

### Альтернативный способ - анализ одного файла:
```python
# Для анализа существующего файла
processor = DatasetProcessor(input_files=[], output_dir="data/processed")
data = processor.load_dataset("existing_file.json")
stats = processor.calculate_stats(data.get('dialogs', []))
```

## 🎯 Операции обработки

### 1. Анализ датасета:
- Подсчет общего количества диалогов
- Определение процента диалогов с ПД
- Анализ распределения типов сущностей
- Вычисление общего количества сущностей

### 2. Объединение датасетов:
```python
# Автоматическое объединение метаданных
merged_metadata = {
    "total_dialogs": stats.total_dialogs,
    "merge_timestamp": datetime.now().timestamp(),
    "source_files": self.input_files,
    "merged_stats": {
        "total_dialogs": stats.total_dialogs,
        "pd_dialogs": stats.pd_dialogs,
        "pd_percentage": stats.pd_percentage,
        "total_entities": stats.total_entities,
        "entity_types": stats.entity_types,
        "total_cost_tokens": total_cost_tokens
    }
}
```

### 3. Разделение на train/val:
- Случайное перемешивание диалогов
- Разделение по заданному соотношению
- Пересчет ID для каждого набора
- Создание отдельных метаданных

### 4. Анализ содержимого:
- Статистика по типам сущностей
- Процент диалогов с персональными данными
- Общее количество токенов
- Распределение по источникам

## 📤 Выходные данные

### Статистика анализа:
```
📊 Статистика датасета: data/generated/dataset.json
Всего диалогов: 491
Диалогов с ПД: 32 (6.5%)
Диалогов без ПД: 459
Всего сущностей: 34

Типы сущностей:
  ADDRESS: 1
  PHONE: 13
  EMAIL: 5
  PERSON: 4
```

### Объединенный датасет:
```json
{
  "metadata": {
    "total_dialogs": 3352,
    "merge_timestamp": 1750283188.0569782,
    "source_files": ["file1.json", "file2.json"],
    "merged_stats": {
      "total_dialogs": 3352,
      "pd_dialogs": 714,
      "pd_percentage": 21.3,
      "total_entities": 1200,
      "entity_types": {
        "PERSON": 400,
        "PHONE": 350,
        "EMAIL": 200
      },
      "total_cost_tokens": 101532
    }
  },
  "dialogs": [...]
}
```

## 🔧 Внутренние компоненты

### Класс DatasetProcessor:
```python
class DatasetProcessor:
    def __init__(self, input_files: List[str], output_dir: str = "data/processed"):
        self.input_files = input_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_dataset(self, file_path: str) -> Dict[str, Any]
    def calculate_stats(self, dialogs: List[Dict[str, Any]]) -> DatasetStats
    def merge_datasets(self) -> Dict[str, Any]
    def split_dataset(self, data: Dict[str, Any], train_ratio: float = 0.7) -> Tuple[Dict[str, Any], Dict[str, Any]]
    def save_dataset(self, data: Dict[str, Any], filename: str)
    def process_all(self, train_ratio: float = 0.7, seed: int = 42)
```

### Класс DatasetStats:
```python
@dataclass
class DatasetStats:
    total_dialogs: int
    pd_dialogs: int
    non_pd_dialogs: int
    pd_percentage: float
    total_entities: int
    entity_types: Dict[str, int]
```

### Функции анализа:
- `analyze_dataset()` - Анализ одного файла с выводом статистики
- `main()` - Обработка аргументов командной строки

## 📦 Зависимости

```txt
json
random
pathlib
typing
dataclasses
logging
argparse
glob
datetime
```

## ⚠️ Примечания

- **Автоматическое создание папок**: Создает выходную папку если не существует
- **Пересчет ID**: Автоматически пересчитывает ID диалогов для уникальности
- **Поддержка glob**: Работает как с развернутыми shell списками, так и с паттернами
- **Статистика**: Подробная статистика на каждом этапе обработки
- **Логирование**: Подробные логи всех операций

## 🔗 Связанные файлы

- `scripts/merge_datasets.py` - Простое слияние датасетов
- `scripts/split_dataset.py` - Разделение датасетов
- `debug/test_pd_detection.py` - Тесты функций обработки 