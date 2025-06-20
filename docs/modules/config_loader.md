# ⚙️ Загрузчик конфигурации - config_loader.py

## 📋 Назначение

Модуль `scripts/config_loader.py` предназначен для централизованной загрузки и валидации конфигурационных параметров из файла `config.env`. Обеспечивает типобезопасность, валидацию и значения по умолчанию.

## 🛠️ Основные функции

- Загрузка конфигурации из .env файлов
- Валидация типов и значений параметров
- Предоставление значений по умолчанию
- Проверка обязательных параметров
- Форматированный вывод конфигурации
- Кэширование загруженных настроек

## 📥 Программный интерфейс

### Основные функции:
```python
from scripts.config_loader import load_dataset_config, ConfigLoader

# Простая загрузка конфигурации
config = load_dataset_config()

# Продвинутая работа с конфигурацией
loader = ConfigLoader(config_file="custom_config.env")
config = loader.load_config()
loader.print_config()
```

### Класс ConfigLoader:
```python
class ConfigLoader:
    def __init__(self, config_file: str = "config.env"):
        self.config_file = config_file
        self._config_cache = None
    
    def load_config(self) -> Dict[str, Any]
    def validate_config(self, config: Dict) -> bool
    def print_config(self, hide_sensitive: bool = True)
    def get_default_config(self) -> Dict[str, Any]
```

## 🎯 Поддерживаемые параметры

### OpenAI API конфигурация:
```python
OPENAI_PARAMS = {
    'OPENAI_API_KEY': {
        'required': True,
        'type': str,
        'description': 'API ключ OpenAI'
    },
    'MODEL': {
        'required': False,
        'type': str,
        'default': 'gpt-4o-mini',
        'description': 'Модель OpenAI для генерации'
    }
}
```

### Параметры генерации датасета:
```python
GENERATION_PARAMS = {
    'TOTAL_DIALOGS': {
        'required': False,
        'type': int,
        'default': 100,
        'min_value': 1,
        'max_value': 10000,
        'description': 'Общее количество диалогов'
    },
    'MAX_BATCH_SIZE': {
        'required': False,
        'type': int,
        'default': 25,
        'min_value': 1,
        'max_value': 100,
        'description': 'Максимальный размер пачки'
    }
}
```

### Режимы работы:
```python
MODE_PARAMS = {
    'PREVIEW_CONFIG': {
        'required': False,
        'type': bool,
        'default': False,
        'description': 'Показать примеры конфигурации'
    },
    'DRY_RUN': {
        'required': False,
        'type': bool,
        'default': False,
        'description': 'Пробный запуск без генерации'
    },
    'ASYNC_GENERATION': {
        'required': False,
        'type': bool,
        'default': True,
        'description': 'Асинхронная генерация'
    }
}
```

## 📤 Формат возвращаемых данных

### Структура конфигурации:
```python
{
    'api_key': 'sk-...',
    'model': 'gpt-4o-mini',
    'total_dialogs': 100,
    'max_batch_size': 25,
    'output_file': '',
    'preview_config': False,
    'dry_run': False,
    'async_generation': True,
    'max_concurrency': 3,
    'base_url': None
}
```

### Вывод конфигурации:
```
⚙️ Загруженная конфигурация:
==================================================
🔑 OpenAI API ключ: ✅ установлен
🤖 Модель: gpt-4o-mini
📊 Всего диалогов: 100
📦 Размер пачки: 25
📁 Выходной файл: авто-генерация
🔍 Режим превью: отключен
🧪 Пробный запуск: отключен
⚡ Асинхронная генерация: включена
🔄 Максимальная параллельность: 3
```

## 🔧 Внутренние компоненты

### Валидация параметров:
```python
def _validate_parameter(self, key: str, value: Any, schema: Dict) -> bool:
    """Валидация отдельного параметра"""
    # Проверка типа
    if not isinstance(value, schema['type']):
        return False
    
    # Проверка диапазона для числовых значений
    if schema['type'] == int:
        if 'min_value' in schema and value < schema['min_value']:
            return False
        if 'max_value' in schema and value > schema['max_value']:
            return False
    
    return True
```

### Преобразование типов:
```python
TYPE_CONVERTERS = {
    'str': str,
    'int': int,
    'float': float,
    'bool': lambda x: x.lower() in ('true', '1', 'yes', 'on')
}
```

### Обработка ошибок:
```python
class ConfigValidationError(Exception):
    """Ошибка валидации конфигурации"""
    pass

class ConfigFileNotFoundError(Exception):
    """Файл конфигурации не найден"""
    pass
```

## 📦 Зависимости

```txt
os
typing
pathlib
```

## 🔒 Безопасность

### Защита чувствительных данных:
```python
SENSITIVE_KEYS = [
    'OPENAI_API_KEY',
    'API_KEY', 
    'SECRET',
    'PASSWORD',
    'TOKEN'
]

def _mask_sensitive_value(self, key: str, value: str) -> str:
    """Маскировка чувствительных значений при выводе"""
    if any(sensitive in key.upper() for sensitive in SENSITIVE_KEYS):
        if len(value) > 8:
            return f"{value[:4]}...{value[-4:]}"
        else:
            return "***"
    return value
```

### Проверка прав доступа:
```python
def _check_file_permissions(self, filepath: str) -> bool:
    """Проверка прав доступа к файлу конфигурации"""
    import stat
    file_stat = os.stat(filepath)
    # Проверяем, что файл не доступен для чтения другим пользователям
    return not (file_stat.st_mode & stat.S_IROTH)
```

## ⚠️ Примечания

- **Типобезопасность**: Автоматическое преобразование и валидация типов
- **Гибкость**: Поддержка пользовательских файлов конфигурации
- **Безопасность**: Маскировка чувствительных данных при выводе
- **Кэширование**: Избегает повторного парсинга файлов
- **Обработка ошибок**: Подробные сообщения об ошибках конфигурации

## 🔗 Связанные файлы

- `config.env` - Основной файл конфигурации
- `config.env.example` - Пример конфигурации
- `scripts/generate_dataset.py` - Использует этот модуль
- `scripts/dataset_config.py` - Дополнительные конфигурации 