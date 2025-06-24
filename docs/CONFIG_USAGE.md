# Использование конфигурации из файла

## Обзор

Скрипт `generate_dataset.py` использует конфигурацию из файла `config.env` для настройки параметров генерации датасетов.

## Структура файла config.env

```bash
# OpenAI API конфигурация
OPENAI_API_KEY=your-api-key-here

# Параметры генерации датасета
TOTAL_DIALOGS=100
MAX_BATCH_SIZE=25
OUTPUT_FILE=
MODEL=gpt-4o-mini

# Режимы работы (true/false)
PREVIEW_CONFIG=false
DRY_RUN=false
```

## Параметры конфигурации

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| `OPENAI_API_KEY` | string | API ключ OpenAI | обязательный |
| `TOTAL_DIALOGS` | integer | Общее количество диалогов | 100 |
| `MAX_BATCH_SIZE` | integer | Максимальный размер пачки | 25 |
| `OUTPUT_FILE` | string | Путь к выходному файлу | авто-генерация |
| `MODEL` | string | Модель OpenAI | gpt-4o-mini |
| `PREVIEW_CONFIG` | boolean | Показать примеры конфигурации | false |
| `DRY_RUN` | boolean | Пробный запуск без генерации | false |

## Способы запуска

### 1. Напрямую с активацией окружения (рекомендуется)
```bash
source venv/bin/activate
python scripts/generate_dataset.py
```

### 2. Тестирование с разными настройками
Отредактируйте `config.env` под ваши нужды:
```bash
nano config.env
```

## Режимы работы

### Обычный режим
Установите в `config.env`:
```bash
PREVIEW_CONFIG=false
DRY_RUN=false
```

### Режим превью конфигурации
Показывает примеры случайных конфигураций:
```bash
PREVIEW_CONFIG=true
DRY_RUN=false
```

### Пробный запуск
Показывает план генерации без выполнения:
```bash
PREVIEW_CONFIG=false
DRY_RUN=true
```

## Файлы модульной архитектуры

- `scripts/config_loader.py` - модуль загрузки конфигурации
- `scripts/generate_dataset.py` - основной скрипт генерации
- `config.env` - файл конфигурации
- `config.env.example` - пример конфигурации

## Преимущества системы конфигурации

1. **Простота использования** - не нужно помнить аргументы командной строки
2. **Повторяемость** - одинаковая конфигурация для всех запусков
3. **Безопасность** - API ключи хранятся в файле, а не в истории команд
4. **Автоматизация** - легко интегрировать в скрипты
5. **Модульность** - отдельные модули для разных функций

## Настройка конфигурации

### Первоначальная настройка:
```bash
# Скопируйте пример
cp config.env.example config.env

# Отредактируйте файл
nano config.env

# Добавьте ваш OpenAI API ключ
OPENAI_API_KEY=your-actual-api-key-here
```

### Для разных проектов:
```bash
# Создайте копии конфигурации
cp config.env config-small.env
cp config.env config-large.env

# Настройте под разные нужды
echo "TOTAL_DIALOGS=50" >> config-small.env
echo "TOTAL_DIALOGS=500" >> config-large.env
```

## Использование в скриптах

```bash
# Использование основной конфигурации
python scripts/generate_dataset.py

# Проверка конфигурации через модуль
python -c "
from scripts.config_loader import load_config
config = load_config()
print(f'Будет создано {config.TOTAL_DIALOGS} диалогов')
"
``` 