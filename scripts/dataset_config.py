"""
Конфигурация для генерации датасета анонимизации ПД
"""
import random
from typing import Dict, List, Any

# Параметры для случайного семплирования
GENERATION_CONFIG = {
    "business_spheres": [
        "автомобили",
        "недвижимость", 
        "медицина",
        "образование",
        "финансы",
        "розничная торговля",
        "услуги",
        "техподдержка"
    ],
    
    "regional_focus": [
        "Москва",
        "Санкт-Петербург",
        "Новосибирск",
        "Екатеринбург",
        "Нижний Новгород",
        "Казань",
        "Ростов-на-Дону"
    ],
    
    "transcription_quality": [
        "отличное",
        "среднее", 
        "плохое"
    ],
    
    "communication_style": [
        "официальный",
        "разговорный",
        "смешанный"
    ],
    
    "pd_percentage_ranges": [
        (70, 80),  # Высокий процент ПД
    ],
    
    "dialog_lengths": [
        (3, 6),    # Короткие диалоги
        (6, 12),   # Средние диалоги
        (12, 20)   # Длинные диалоги
    ],
    
    "batch_sizes": [25, 30, 35, 40, 50]
}

# Веса для более реалистичного распределения
WEIGHTED_CONFIG = {
    "business_spheres": {
        "автомобили": 0.2,
        "недвижимость": 0.2,
        "медицина": 0.15,
        "финансы": 0.15,
        "розничная торговля": 0.1,
        "образование": 0.1,
        "услуги": 0.05,
        "техподдержка": 0.05
    },
    
    "regional_focus": {
        "Москва": 0.3,
        "Санкт-Петербург": 0.25,
        "Новосибирск": 0.15,
        "Екатеринбург": 0.1,
        "Нижний Новгород": 0.08,
        "Казань": 0.07,
        "Ростов-на-Дону": 0.05
    },
    
    "transcription_quality": {
        "среднее": 0.45,
        "плохое": 0.25,
        "отличное": 0.3
    },
    
    "communication_style": {
        "смешанный": 0.5,
        "разговорный": 0.3,
        "официальный": 0.2
    }
}

def sample_random_config() -> Dict[str, Any]:
    """
    Создает случайную конфигурацию для генерации диалогов
    
    Returns:
        Словарь с параметрами генерации
    """
    # Используем взвешенный выбор для более реалистичного распределения
    business_sphere = weighted_choice(WEIGHTED_CONFIG["business_spheres"])
    regional_focus = weighted_choice(WEIGHTED_CONFIG["regional_focus"])
    transcription_quality = weighted_choice(WEIGHTED_CONFIG["transcription_quality"])
    communication_style = weighted_choice(WEIGHTED_CONFIG["communication_style"])
    
    # Случайный процент ПД
    pd_range = random.choice(GENERATION_CONFIG["pd_percentage_ranges"])
    pd_percentage = random.randint(pd_range[0], pd_range[1])
    
    # Случайный размер пачки
    batch_size = random.choice(GENERATION_CONFIG["batch_sizes"])
    
    return {
        "business_sphere": business_sphere,
        "regional_focus": regional_focus,
        "transcription_quality": transcription_quality,
        "communication_style": communication_style,
        "pd_percentage": pd_percentage,
        "false_positive_percentage": random.randint(10, 15),
        "batch_size": batch_size
    }

def weighted_choice(weights_dict: Dict[str, float]) -> str:
    """
    Делает взвешенный случайный выбор из словаря
    
    Args:
        weights_dict: Словарь {значение: вес}
        
    Returns:
        Выбранное значение
    """
    choices = list(weights_dict.keys())
    weights = list(weights_dict.values())
    return random.choices(choices, weights=weights)[0]

def generate_batch_configs(total_dialogs: int, max_batch_size: int = 50) -> List[Dict[str, Any]]:
    """
    Генерирует список конфигураций для пачек диалогов
    
    Args:
        total_dialogs: Общее количество диалогов
        max_batch_size: Максимальный размер пачки
        
    Returns:
        Список конфигураций для каждой пачки
    """
    configs = []
    remaining_dialogs = total_dialogs
    
    while remaining_dialogs > 0:
        config = sample_random_config()
        
        # Ограничиваем размер пачки оставшимся количеством диалогов
        actual_batch_size = min(config["batch_size"], remaining_dialogs, max_batch_size)
        config["batch_size"] = actual_batch_size
        
        configs.append(config)
        remaining_dialogs -= actual_batch_size
    
    return configs

# Системные настройки
OPENAI_CONFIG = {
    "model": "gpt-4o-mini",  # Более дешевая модель для массовой генерации
    "max_tokens": 4000,
    "temperature": 0.3,
    "timeout": 120
}

# Параметры генерации по умолчанию
DEFAULT_GENERATION_PARAMS = {
    "total_dialogs": 100,
    "max_batch_size": 20,
    "async_generation": True,
    "max_concurrency": 3
} 