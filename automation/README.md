# Automation Scripts

Эта папка содержит bash-обертки для автоматизации основных операций с датасетами.

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