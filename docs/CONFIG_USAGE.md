# Использование конфигурации из файла

## Обзор

Скрипт `generate_dataset.py` был изменен для использования конфигурации из файла `config.env` вместо аргументов командной строки.

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

### 1. Через bash скрипт (рекомендуется)
```bash
./run_generator.sh
```

### 2. Напрямую с активацией окружения
```bash
source venv/bin/activate
python generate_dataset.py
```

### 3. Тестирование конфигурации
```bash
source venv/bin/activate
python test_config.py
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
- `test_config.py` - тестирование конфигурации
- `run_generator.sh` - bash скрипт запуска
- `config.env` - файл конфигурации

## Преимущества новой системы

1. **Простота использования** - не нужно помнить аргументы командной строки
2. **Повторяемость** - одинаковая конфигурация для всех запусков
3. **Безопасность** - API ключи хранятся в файле, а не в истории команд
4. **Автоматизация** - легко интегрировать в CI/CD
5. **Модульность** - отдельные модули для разных функций 

## Изменения в generate_dataset.py

Скрипт `scripts/generate_dataset.py` был изменен для использования конфигурации из файла `config.env` вместо аргументов командной строки. 

```bash
python scripts/generate_dataset.py
``` 