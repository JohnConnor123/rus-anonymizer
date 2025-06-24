# 🔒 Anonymizer - Система анонимизации персональных данных

Комплексная система для анонимизации персональных данных в русскоязычных текстах с использованием различных подходов: от регулярных выражений до современных нейронных сетей.

## 📁 Структура проекта

```
anonymizer/
├── 📂 anonymizers/           # Основные анонимизаторы
│   ├── base.py              # Базовый класс для всех анонимизаторов
│   ├── regexp_baseline/     # Базовый анонимизатор на regex
│   ├── natasha_enhanced/    # Анонимизатор на основе Natasha
│   ├── deeppavlov_ner/      # Анонимизатор на основе DeepPavlov
│   ├── spacy_extended/      # Анонимизатор на основе SpaCy
│   ├── transformers_bert/   # Анонимизатор на основе Transformers
│   └── hybrid_advanced/     # Гибридный анонимизатор
│
├── 📂 scripts/              # Основные скрипты проекта
│   ├── generate_dataset.py  # Генерация синтетических данных
│   ├── validate_simple.py   # Валидация всех анонимизаторов
│   ├── validate_deeppavlov.py # Специализированная валидация DeepPavlov
│   ├── config_loader.py     # Загрузчик конфигурации
│   └── dataset_*.py         # Модули генерации данных
│
├── 📂 tests/                # Тесты
│   ├── test_dataset_generator.py # Тесты генератора
│   ├── test_data_processor.py    # Тесты обработчика данных
│   └── test_*.py                 # Другие тесты
│
├── 📂 docs/                 # Документация
│   ├── README.md            # Навигация по документации
│   ├── CONFIG_USAGE.md      # Использование конфигурации
│   ├── reference/           # Справочные материалы
│   ├── scripts/             # Документация скриптов
│   └── automation/          # Документация автоматизации
│
├── 📂 data/                # Данные и результаты
│   ├── generated/          # Сгенерированные датасеты
│   ├── processed/          # Обработанные данные
│   └── reports/            # Отчеты валидации
│
├── 📂 automation/          # Скрипты автоматизации
├── 📂 venv/                # Основное виртуальное окружение
├── 📂 venv-deeppavlov/     # Окружение для DeepPavlov
│
├── requirements.txt        # Зависимости для основного окружения
├── requirements-deeppavlov.txt  # Зависимости для DeepPavlov
├── pyproject.toml          # Конфигурация проекта
└── README.md              # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Основное окружение
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Для DeepPavlov (отдельное окружение)
python -m venv venv-deeppavlov
source venv-deeppavlov/bin/activate
pip install -r requirements-deeppavlov.txt
python -m deeppavlov install ner_rus_bert
```

### 2. Базовое использование

```python
from anonymizers.natasha_enhanced.natasha_anonymizer import NatashaAnonymizer

# Создаем анонимизатор
anonymizer = NatashaAnonymizer(aggressiveness=0.8)

# Анонимизируем текст
text = "Меня зовут Иван Петров, мой телефон +7-999-123-45-67"
result = anonymizer.anonymize(text)
print(result)  # "Меня зовут <PERSON>, мой телефон <PHONE>"
```

### 3. Валидация всех анонимизаторов

```bash
# Активируйте нужное окружение
source venv/bin/activate

# Запустите валидацию
python scripts/validate_simple.py --dataset data/processed/merged_dataset.json
```

## 📊 Сравнение анонимизаторов (актуальные результаты)

| Анонимизатор | F1 Micro | F1 Macro | Precision | Recall | Скорость | Рекомендация |
|-------------|----------|-----------|-----------|--------|----------|--------------|
| **RegExp Baseline** | **0.499** | **0.275** | **0.654** | **0.403** | Очень быстрая | 🏆 **Лучший общий** |
| **Natasha Enhanced** | 0.237 | 0.060 | 0.351 | 0.179 | Быстрая | Для русских имен |
| **Hybrid Advanced** | 0.236 | 0.066 | 0.377 | 0.172 | Средняя | Для сложных случаев |
| **SpaCy Extended** | 0.225 | 0.074 | 0.307 | 0.177 | Быстрая | Универсальный |
| **DeepPavlov NER** | 0.205 | 0.056 | 0.273 | 0.165 | Медленная | Экспериментальный |
| **Transformers BERT** | 0.190 | 0.042 | 0.246 | 0.156 | Медленная | В разработке |

### 📈 Интерпретация метрик
- **F1 Micro** - общее качество с учетом количества сущностей каждого типа
- **F1 Macro** - среднее качество по всем типам сущностей (без взвешивания)
- **Низкий F1 Macro** у всех методов указывает на плохую работу с редкими типами ПД

## 🎯 Рекомендации по выбору

- **Максимальное качество**: `RegExpBaselineAnonymizer` - лучший F1-Score и precision
- **Для русских имен**: `NatashaAnonymizer` - хорошо находит PERSON и ORGANIZATION  
- **Универсальность**: `SpacyAnonymizer` - стабильная работа
- **Экспериментальные**: `HybridAnonymizer`, `DeepPavlovAnonymizer` - требуют доработки

## 🔧 Настройка агрессивности

```python
# Консервативная анонимизация (меньше ложных срабатываний)
anonymizer = NatashaAnonymizer(aggressiveness=0.3)

# Агрессивная анонимизация (больше покрытие)
anonymizer = NatashaAnonymizer(aggressiveness=0.9)
```

## 📝 Поддерживаемые типы ПД

- **Базовые**: Имена, телефоны, email, адреса
- **Документы**: Паспорта, СНИЛС, ИНН, банковские карты
- **Персональные**: Возраст, дни рождения, семейное положение
- **Профессиональные**: Должности, места работы, образование
- **Медицинские**: Группы крови, аллергии, полисы ОМС
- **Транспортные**: Автомобильные номера
- **Технические**: IP-адреса, логины

## 🧪 Тестирование

```bash
# Быстрая валидация всех анонимизаторов
python scripts/validate_simple.py

# Валидация с конкретным датасетом
python scripts/validate_simple.py --dataset data/processed/your_dataset.json

# Для DeepPavlov используйте отдельное окружение
source venv-deeppavlov/bin/activate
python scripts/validate_deeppavlov.py --dataset data/processed/merged_dataset.json
```

## 📚 Документация

- [Руководство по использованию](docs/reference/USAGE.md)
- [Описание моделей](docs/reference/MODELS.md)
- [Результаты бенчмарков](docs/reference/BENCHMARKS.md)
- [Настройка конфигурации](docs/CONFIG_USAGE.md)

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Добавьте тесты
4. Отправьте pull request

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей. 