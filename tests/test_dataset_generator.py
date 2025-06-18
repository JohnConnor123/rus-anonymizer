#!/usr/bin/env python3
"""
Тест нового генератора датасета
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.dataset_config import sample_random_config, generate_batch_configs
    print("✅ Импорт scripts.dataset_config успешен")
    
    # Тестируем случайную конфигурацию
    config = sample_random_config()
    print("\n🎲 Пример случайной конфигурации:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Тестируем генерацию конфигураций пачек
    batch_configs = generate_batch_configs(total_dialogs=100, max_batch_size=25)
    print(f"\n📦 Сгенерировано {len(batch_configs)} пачек для 100 диалогов")
    
    for i, batch_config in enumerate(batch_configs[:3]):  # Показываем первые 3
        print(f"\n   Пачка {i+1}:")
        print(f"     Размер: {batch_config['batch_size']}")
        print(f"     Сфера: {batch_config['business_sphere']}")
        print(f"     Регион: {batch_config['regional_focus']}")
        print(f"     ПД: {batch_config['pd_percentage']}%")

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
except Exception as e:
    print(f"❌ Ошибка: {e}")

try:
    from scripts.dataset_prompts import format_prompt
    print("\n✅ Импорт scripts.dataset_prompts успешен")
    
    # Тестируем форматирование промпта
    test_config = {
        "total_dialogs": 5,
        "business_sphere": "автомобили",
        "regional_focus": "Москва",
        "transcription_quality": "среднее",
        "communication_style": "смешанный",
        "pd_percentage": 75,
        "false_positive_percentage": 15,
        "batch_size": 5
    }
    
    prompt = format_prompt(test_config)
    print(f"\n📝 Промпт сгенерирован, длина: {len(prompt)} символов")
    print(f"   Первые 200 символов: {prompt[:200]}...")

except ImportError as e:
    print(f"❌ Ошибка импорта scripts.dataset_prompts: {e}")
except Exception as e:
    print(f"❌ Ошибка scripts.dataset_prompts: {e}")

print("\n�� Тест завершен!") 