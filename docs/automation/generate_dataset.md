# 🚀 Автоматизированная генерация - generate_dataset.sh

## 📋 Назначение

Bash скрипт `automation/generate_dataset.sh` предназначен для полностью автоматизированной генерации датасетов с проверками окружения, активацией виртуального окружения и обработкой ошибок.

## 🛠️ Основные функции

- Автоматическая активация виртуального окружения
- Проверка наличия необходимых зависимостей
- Валидация конфигурационного файла
- Запуск генерации с логированием
- Обработка ошибок и автовосстановление
- Создание резервных копий

## 📥 Способы запуска

### Простой запуск:
```bash
./automation/generate_dataset.sh
```

### С дополнительными параметрами:
```bash
# Указание конфигурационного файла
./automation/generate_dataset.sh --config custom_config.env

# Тихий режим (без интерактивных подтверждений)
./automation/generate_dataset.sh --quiet

# Режим отладки  
./automation/generate_dataset.sh --debug

# Указание выходной папки
./automation/generate_dataset.sh --output-dir /path/to/output
```

### Планировщик cron:
```bash
# Запуск каждый день в 2:00
0 2 * * * /path/to/anonymizer/automation/generate_dataset.sh --quiet
```

## 🔧 Внутренний алгоритм

### 1. Предварительные проверки:
```bash
# Проверка окружения
check_prerequisites() {
    check_python_version
    check_virtual_env
    check_config_file
    check_disk_space
    check_api_connectivity
}
```

### 2. Активация окружения:
```bash
# Поиск и активация venv
activate_environment() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        create_virtual_environment
    fi
}
```

### 3. Запуск генерации:
```bash
# Основной процесс
run_generation() {
    validate_config
    backup_previous_data
    python scripts/generate_dataset.py
    verify_output
    cleanup_temp_files
}
```

### 4. Обработка ошибок:
```bash
# Автовосстановление
handle_errors() {
    case $ERROR_CODE in
        1) retry_with_smaller_batch ;;
        2) check_api_limits ;;
        3) restore_from_backup ;;
        *) send_notification ;;
    esac
}
```

## 📊 Параметры и опции

| Опция | Описание | По умолчанию |
|-------|----------|--------------|
| `--config` | Путь к config.env | ./config.env |
| `--output-dir` | Папка для результатов | ./data/generated |
| `--backup-dir` | Папка для резервных копий | ./data/backups |
| `--quiet` | Тихий режим | false |
| `--debug` | Режим отладки | false |
| `--no-backup` | Без создания резервных копий | false |
| `--force` | Игнорировать предупреждения | false |

## 📤 Выходные данные

### Успешное выполнение:
```
🚀 Запуск автоматизированной генерации датасета
==================================================
✅ Проверка окружения: OK
✅ Активация venv: OK  
✅ Валидация конфигурации: OK
✅ Проверка API подключения: OK

🎯 Начинаю генерацию...
📊 Сгенерировано диалогов: 100
💾 Файл сохранен: data/generated/dataset_20240120_153045.json
🔒 Резервная копия: data/backups/backup_20240120_153045.tar.gz

⏱️  Время выполнения: 5 минут 23 секунды
✅ Генерация завершена успешно!
```

### Обработка ошибок:
```
❌ Ошибка: Недостаточно места на диске
💡 Решение: Освободите минимум 1GB места
🔄 Повторная попытка через 30 секунд...

❌ Ошибка: API лимит превышен  
💡 Решение: Уменьшаю размер пачки с 25 до 10
🔄 Перезапускаю генерацию...
```

## 🔧 Настройка окружения

### Требования к системе:
```bash
# Минимальные требования
PYTHON_VERSION="3.8+"
DISK_SPACE="1GB"
RAM="2GB" 
NETWORK="Stable internet connection"
```

### Переменные окружения:
```bash
# Дополнительные настройки
export GENERATION_TIMEOUT=3600        # Таймаут генерации (сек)
export MAX_RETRIES=3                  # Максимум попыток
export BACKUP_RETENTION_DAYS=7        # Хранение резервных копий
export LOG_LEVEL="INFO"               # Уровень логирования
```

### Создание логов:
```bash
# Структура логирования  
logs/
├── generate_dataset_20240120.log     # Основной лог
├── api_requests.log                  # Логи API запросов
├── errors.log                        # Логи ошибок
└── performance.log                   # Метрики производительности
```

## 🔒 Безопасность

### Защита API ключей:
```bash
# Проверка прав доступа к config.env
secure_config() {
    chmod 600 config.env
    check_file_permissions config.env
}
```

### Ротация логов:
```bash  
# Автоматическая очистка старых логов
cleanup_logs() {
    find logs/ -name "*.log" -mtime +30 -delete
    compress_old_logs
}
```

## ⚠️ Примечания

- **Автономность**: Работает без участия пользователя
- **Надежность**: Автоматическое восстановление при ошибках
- **Масштабируемость**: Подходит для регулярного запуска
- **Мониторинг**: Подробные логи для диагностики
- **Интеграция**: Легко интегрируется с CI/CD пайплайнами

## 🔗 Связанные файлы

- `scripts/generate_dataset.py` - Основной скрипт генерации
- `config.env` - Файл конфигурации
- `automation/validate_dataset.sh` - Валидация результатов
- `logs/` - Папка с логами выполнения 