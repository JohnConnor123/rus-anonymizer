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
├── 📂 tests/                # Тесты и валидация
│   ├── test_dataset.json    # Тестовый датасет
│   ├── validate_simple.py   # Простая валидация (основной)
│   └── legacy/              # Старые тесты
│
├── 📂 docs/                 # Документация
│   ├── USAGE.md            # Руководство по использованию
│   ├── MODELS.md           # Описание моделей
│   └── BENCHMARKS.md       # Результаты бенчмарков
│
├── 📂 data/                # Данные и результаты
│   ├── results/            # Результаты валидации
│   └── samples/            # Примеры текстов
│
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
source venv/bin/activate  # или source venv-deeppavlov/bin/activate

# Запустите валидацию
cd tests/
python validate_simple.py
```

## 📊 Сравнение анонимизаторов

| Анонимизатор | F1-Score | Precision | Recall | Скорость | Рекомендация |
|-------------|----------|-----------|--------|----------|--------------|
| **Hybrid Advanced** | 0.85 | 0.82 | 0.88 | Средняя | 🏆 **Лучший общий** |
| **DeepPavlov NER** | 0.47 | 0.63 | 0.38 | Медленная | Для документов |
| **Natasha Enhanced** | 0.75 | 0.78 | 0.72 | Быстрая | Для имен/мест |
| **SpaCy Extended** | 0.68 | 0.71 | 0.65 | Быстрая | Универсальный |
| **RegExp Baseline** | 0.45 | 0.89 | 0.31 | Очень быстрая | Для простых задач |

## 🎯 Рекомендации по выбору

- **Максимальное качество**: `HybridAnonymizer` - комбинирует все подходы
- **Скорость + качество**: `NatashaAnonymizer` - оптимальный баланс  
- **Высокая точность**: `RegExpBaselineAnonymizer` - для структурированных данных
- **Универсальность**: `SpacyAnonymizer` - хорошо для разных типов текстов

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
# Быстрая валидация
python tests/validate_simple.py

# Полное тестирование
python -m pytest tests/

# Для DeepPavlov используйте отдельное окружение
source venv-deeppavlov/bin/activate
python tests/validate_simple.py
```

## 📚 Документация

- [Руководство по использованию](docs/USAGE.md)
- [Описание моделей](docs/MODELS.md)
- [Результаты бенчмарков](docs/BENCHMARKS.md)

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Добавьте тесты
4. Отправьте pull request

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей. 