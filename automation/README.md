# Automation Scripts

Эта папка предназначена для bash-скриптов автоматизации основных операций с датасетами.

## 📋 Текущее состояние

В настоящее время автоматизация выполняется через Python скрипты в папке `scripts/`. Bash-обертки могут быть добавлены в будущем для упрощения работы.

## 🚀 Альтернативы автоматизации

### Генерация датасетов
```bash
# Прямой запуск с конфигурацией
source venv/bin/activate
python scripts/generate_dataset.py
```

### Валидация анонимизаторов
```bash
# Основная валидация
source venv/bin/activate
python scripts/validate_simple.py --dataset data/processed/merged_dataset.json

# DeepPavlov валидация
source venv-deeppavlov/bin/activate  
python scripts/validate_deeppavlov.py --dataset data/processed/merged_dataset.json
```

### Обработка данных
```bash
# Объединение датасетов (если есть соответствующий скрипт)
source venv/bin/activate
python scripts/data_processor.py --input data/generated/*.json --output data/processed/merged.json
```

## 🔧 Создание bash-оберток

Для удобства можно создать простые bash-скрипты:

### generate_dataset.sh (пример)
```bash
#!/bin/bash
set -e

# Переход в корневую папку
cd "$(dirname "$0")/.."

# Активация окружения
source venv/bin/activate

# Проверка конфигурации
if [ ! -f config.env ]; then
    echo "❌ Файл config.env не найден. Скопируйте config.env.example"
    exit 1
fi

# Запуск генерации
echo "🎲 Запуск генерации датасета..."
python scripts/generate_dataset.py

echo "✅ Генерация завершена. Проверьте папку data/generated/"
```

### validate_dataset.sh (пример)
```bash
#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

DATASET=${1:-"data/processed/merged_dataset.json"}

if [ ! -f "$DATASET" ]; then
    echo "❌ Датасет $DATASET не найден"
    exit 1
fi

echo "🔍 Запуск валидации анонимизаторов..."
python scripts/validate_simple.py --dataset "$DATASET"

echo "✅ Валидация завершена. Результаты в data/reports/"
```

## 📊 Полный workflow автоматизации

```bash
# 1. Настройка
cp config.env.example config.env
nano config.env  # добавьте API ключ

# 2. Генерация данных
source venv/bin/activate
python scripts/generate_dataset.py

# 3. Валидация
python scripts/validate_simple.py --dataset data/generated/llama-4-maverick-*.json

# 4. Просмотр результатов
ls data/reports/
```

## 💡 Рекомендации

- **Используйте Python скрипты напрямую** - они уже имеют все необходимые параметры
- **Создавайте bash-обертки по необходимости** - для часто выполняемых операций
- **Настройте config.env один раз** - для повторного использования
- **Проверяйте виртуальные окружения** - venv для основных скриптов, venv-deeppavlov для DeepPavlov

## 🔍 Планируемые улучшения

В будущих версиях могут быть добавлены:
- Bash-скрипты для автоматизации полного цикла
- Интеграция с CI/CD системами
- Скрипты мониторинга качества анонимизации
- Автоматическое создание отчетов

## 📋 Доступные скрипты:

### 🚀 `generate_dataset.sh`
Генерация нового датасета с помощью OpenAI API.
```bash
./automation/generate_dataset.sh
```
Генерирует датасет в стандартном формате с метаданными:
```json
{
  "metadata": {
    "total_dialogs": 100,
    "generation_timestamp": 1234567890,
    "model_used": "meta-llama/llama-4-maverick",
    "generation_stats": {...}
  },
  "dialogs": [...]
}
```

### 🔍 `validate_dataset.sh`
Валидация анонимизаторов на датасете.
```bash
# Использовать стандартный датасет
./automation/validate_dataset.sh

# Использовать конкретный датасет
./automation/validate_dataset.sh data/generated/my_dataset.json
```

### ✏️ `annotate_dataset.sh`
Шаблон для ручной аннотации датасета.
```bash
./automation/annotate_dataset.sh data/generated/new_dataset.json
```

### 📊 `analyze_quality.sh`
Анализ качества аннотации датасета.
```bash
./automation/analyze_quality.sh data/generated/annotated_dataset.json
```

## 🔧 Настройка:

1. Убедитесь что скрипты исполняемые:
```bash
chmod +x automation/*.sh
```

2. Все скрипты автоматически:
   - Переходят в корневую папку проекта
   - Активируют виртуальное окружение `venv`
   - Проверяют зависимости
   - Выводят подробную информацию о процессе

## 🎯 Преимущества:

- ✅ Автоматическая активация виртуального окружения
- ✅ Проверка зависимостей и конфигурации
- ✅ Единообразный интерфейс для всех операций
- ✅ Подробный вывод с эмодзи для удобства
- ✅ Обработка ошибок с понятными сообщениями
- ✅ Единый стандартный формат датасетов

## 💡 Для разработчиков:

Все скрипты используют `set -e` для остановки при первой ошибке и автоматически определяют корневую папку проекта через `$(dirname "$0")/..`. 