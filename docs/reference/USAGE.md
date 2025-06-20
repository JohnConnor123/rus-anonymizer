# 📖 Руководство по использованию Anonymizer

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Клонирование репозитория
git clone <repository-url>
cd anonymizer

# Создание основного окружения
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Для DeepPavlov (отдельное окружение)
python -m venv venv-deeppavlov
source venv-deeppavlov/bin/activate
pip install -r requirements-deeppavlov.txt
python -m deeppavlov install ner_rus_bert
```

### Настройка конфигурации

```bash
# Скопируйте пример конфигурации
cp config.env.example config.env

# Отредактируйте config.env, добавив ваш API ключ
nano config.env
```

## 🎯 Основные способы использования

### 1. Базовая анонимизация

```python
from anonymizers.natasha_enhanced.natasha_anonymizer import NatashaAnonymizer

# Создание анонимизатора
anonymizer = NatashaAnonymizer(aggressiveness=0.8)

# Анонимизация текста
text = "Меня зовут Иван Петров, мой телефон +7-999-123-45-67, email ivan@example.com"
result = anonymizer.anonymize(text)
print(result)
# Результат: "Меня зовут <PERSON>, мой телефон <PHONE>, email <EMAIL>"
```

### 2. Различные типы анонимизаторов

```python
# RegExp анонимизатор (быстрый, для простых случаев)
from anonymizers.regexp_baseline.regexp_anonymizer import RegExpBaselineAnonymizer
anonymizer = RegExpBaselineAnonymizer()

# SpaCy анонимизатор (универсальный)  
from anonymizers.spacy_extended.spacy_anonymizer import SpacyAnonymizer
anonymizer = SpacyAnonymizer()

# Гибридный анонимизатор (лучшее качество)
from anonymizers.hybrid_advanced.hybrid_anonymizer import HybridAnonymizer
anonymizer = HybridAnonymizer()

# Transformers BERT (для сложных случаев)
from anonymizers.transformers_bert.transformers_anonymizer import TransformersAnonymizer
anonymizer = TransformersAnonymizer()
```

### 3. Настройка агрессивности

```python
# Консервативная анонимизация (меньше ложных срабатываний)
anonymizer = NatashaAnonymizer(aggressiveness=0.3)

# Умеренная анонимизация (рекомендуется)
anonymizer = NatashaAnonymizer(aggressiveness=0.6)

# Агрессивная анонимизация (максимальное покрытие)
anonymizer = NatashaAnonymizer(aggressiveness=0.9)
```

### 4. Пользовательские замены

```python
anonymizer = NatashaAnonymizer(
    custom_replacements={
        'PERSON': '[ИМЯ]',
        'PHONE': '[ТЕЛЕФОН]',
        'EMAIL': '[ПОЧТА]',
        'LOCATION': '[МЕСТО]'
    }
)
```

## 🛠️ Использование скриптов

### Анонимизация из командной строки

```bash
# Анонимизация текста через pipe
echo "Привет, меня зовут Петр Иванов" | python scripts/anonymize.py

# Анонимизация файла
python scripts/anonymize.py --input input.txt --output output.txt

# Выбор конкретного анонимизатора
python scripts/anonymize.py --anonymizer natasha --aggressiveness 0.8
```

### Генерация тестовых данных

```bash
# Настройте config.env и запустите генерацию
source venv/bin/activate
python scripts/generate_dataset.py

# Объединение нескольких датасетов
python scripts/merge_datasets.py --input data/generated/*.json --train-ratio 0.7
```

### Валидация и тестирование

```bash
# Быстрая валидация всех анонимизаторов
python tests/validate_simple.py --dataset data/processed/merged_dataset.json

# Валидация DeepPavlov (отдельное окружение)
source venv-deeppavlov/bin/activate
python tests/validate_deeppavlov_improved.py --dataset data/processed/merged_dataset.json --max-examples 50
```

## 📊 Анализ результатов

### Просмотр метрик

```bash
# После валидации результаты сохраняются в:
ls data/reports/

# Открытие CSV отчетов
python -c "
import pandas as pd
df = pd.read_csv('data/reports/merged_dataset_summary_all_methods.csv', sep=';')
print(df.head())
"
```

### Интерпретация метрик

- **Precision** - точность (доля правильно найденных среди всех найденных)
- **Recall** - полнота (доля найденных среди всех существующих)
- **F1-Score** - гармоническое среднее точности и полноты
- **Accuracy** - общая точность классификации

## 🎛️ Продвинутые настройки

### Кастомные типы сущностей

```python
from anonymizers.base import BaseAnonymizer

class CustomAnonymizer(BaseAnonymizer):
    def __init__(self):
        super().__init__()
        self.entity_patterns = {
            'CUSTOM_ID': r'\b[A-Z]{2}\d{6}\b',  # Пример: AB123456
            'BANK_CARD': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
    
    def anonymize(self, text):
        # Ваша логика анонимизации
        return self._apply_patterns(text)
```

### Интеграция с API

```python
import requests

def anonymize_via_api(text, anonymizer_type='natasha'):
    """Пример API интеграции"""
    # Здесь может быть ваш API endpoint
    response = requests.post('/api/anonymize', {
        'text': text,
        'anonymizer': anonymizer_type,
        'aggressiveness': 0.8
    })
    return response.json()['anonymized_text']
```

## 🚨 Частые проблемы и решения

### Проблема: "ModuleNotFoundError"
```bash
# Убедитесь, что активировано правильное окружение
source venv/bin/activate  # или venv-deeppavlov/bin/activate для DeepPavlov

# Переустановите зависимости
pip install -r requirements.txt
```

### Проблема: Низкое качество анонимизации
```python
# Попробуйте гибридный анонимизатор
from anonymizers.hybrid_advanced.hybrid_anonymizer import HybridAnonymizer
anonymizer = HybridAnonymizer()

# Или увеличьте агрессивность
anonymizer = NatashaAnonymizer(aggressiveness=0.9)
```

### Проблема: Медленная работа
```python
# Используйте RegExp для простых случаев
from anonymizers.regexp_baseline.regexp_anonymizer import RegExpBaselineAnonymizer
fast_anonymizer = RegExpBaselineAnonymizer()

# Или ограничьте размер батча
anonymizer.batch_size = 50
```

## 📈 Лучшие практики

1. **Выбор анонимизатора:**
   - Для скорости: `RegExpBaseline`
   - Для качества: `HybridAdvanced`
   - Для баланса: `NatashaEnhanced`

2. **Настройка агрессивности:**
   - 0.3-0.5: Консервативно (мало ложных срабатываний)
   - 0.6-0.8: Оптимально (баланс качества и скорости)
   - 0.9+: Агрессивно (максимальное покрытие)

3. **Валидация:**
   - Всегда тестируйте на ваших данных
   - Используйте метрики для выбора оптимального анонимизатора
   - Сохраняйте отчеты для анализа трендов

4. **Производственное использование:**
   - Используйте batch-обработку для больших объемов
   - Настройте мониторинг метрик качества
   - Регулярно обновляйте модели и паттерны

## 📚 Дополнительные ресурсы

- [Описание моделей](MODELS.md)
- [Результаты бенчмарков](BENCHMARKS.md)
- [Настройка конфигурации](CONFIG_USAGE.md)
- [Улучшения метрик](METRICS_IMPROVEMENTS.md)
