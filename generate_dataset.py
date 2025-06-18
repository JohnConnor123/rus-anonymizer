#!/usr/bin/env python3
"""
Основной скрипт для генерации датасета анонимизации ПД через OpenAI API
Конфигурация загружается из файла config.env
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Импорты из модулей в папке scripts
from scripts.dataset_openai_generator import OpenAIDatasetGenerator
from scripts.dataset_config import sample_random_config
from scripts.config_loader import load_dataset_config, ConfigLoader


def main():
    """Основная функция генерации датасета"""
    
    try:
        # Загружаем конфигурацию из файла
        config = load_dataset_config()
        loader = ConfigLoader()
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        print("💡 Создайте файл config.env с необходимыми параметрами")
        return 1
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return 1
    
    # Показать пример конфигурации
    if config['preview_config']:
        print("🎲 Пример случайной конфигурации:")
        for i in range(3):
            sample_config = sample_random_config()
            print(f"\n  Вариант {i+1}:")
            for key, value in sample_config.items():
                print(f"    {key}: {value}")
        return
    
    # Проверяем API ключ
    if not config['api_key'] or config['api_key'] == 'your-api-key-here':
        print("❌ Ошибка: не найден OpenAI API ключ!")
        print("💡 Установите корректный OPENAI_API_KEY в файле config.env")
        return 1
    
    # Определяем выходной файл
    if config['output_file']:
        output_file = config['output_file']
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/generated/dataset_{timestamp}.json"
    
    # Создаем директорию если не существует
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Генератор датасета анонимизации ПД")
    print("=" * 50)
    
    # Показываем загруженную конфигурацию
    loader.print_config()
    
    print(f"\n📊 Параметры генерации:")
    print(f"   Диалогов: {config['total_dialogs']}")
    print(f"   Размер пачки: {config['max_batch_size']}")
    print(f"   Модель: {config['model']}")
    print(f"   Выходной файл: {output_file}")
    print(f"   API ключ: {'✅ установлен' if config['api_key'] else '❌ отсутствует'}")
    
    if config['dry_run']:
        print("\n🔍 Это пробный запуск - генерация не будет выполнена")
        
        # Показываем план пачек
        from scripts.dataset_config import generate_batch_configs
        batch_configs = generate_batch_configs(
            config['total_dialogs'], 
            config['max_batch_size']
        )
        
        print(f"\n📦 План генерации ({len(batch_configs)} пачек):")
        for i, batch_config in enumerate(batch_configs):
            print(f"   Пачка {i+1}: {batch_config['batch_size']} диалогов")
            print(f"     Сфера: {batch_config['business_sphere']}")
            print(f"     Регион: {batch_config['regional_focus']}")
            print(f"     Стиль: {batch_config['communication_style']}")
            print(f"     Качество: {batch_config['transcription_quality']}")
            print(f"     ПД: {batch_config['pd_percentage']}%")
            print()
            
        return 0
    
    # Запускаем генерацию
    try:
        print("\n🎯 Начинаю генерацию...")
        
        generator = OpenAIDatasetGenerator(
            api_key=config['api_key'], 
            model=config['model'],
            base_url=config.get('base_url')
        )
        
        if config.get('async_generation', True):
            import asyncio
            stats = asyncio.run(
                generator.generate_dataset_async(
                    total_dialogs=config['total_dialogs'],
                    max_batch_size=config['max_batch_size'],
                    save_path=output_file,
                    max_concurrency=config.get('max_concurrency', 3)
                )
            )
        else:
            stats = generator.generate_dataset(
                total_dialogs=config['total_dialogs'],
                max_batch_size=config['max_batch_size'],
                save_path=output_file
            )
        
        print("\n🎉 Генерация завершена успешно!")
        print("=" * 50)
        print(f"📄 Файл: {stats['save_path']}")
        print(f"📊 Всего диалогов: {stats['total_dialogs']}")
        print(f"✅ С ПД: {stats['with_pd']} ({stats['with_pd']/stats['total_dialogs']*100:.1f}%)")
        print(f"📝 Без ПД: {stats['without_pd']} ({stats['without_pd']/stats['total_dialogs']*100:.1f}%)")
        print(f"💰 Потрачено токенов: {stats['total_cost_tokens']:,}")
        
        if stats['sphere_distribution']:
            print(f"\n🏢 Распределение по сферам:")
            for sphere, count in stats['sphere_distribution'].items():
                print(f"   {sphere}: {count}")
        
        # Показываем пример диалога
        try:
            import json
            with open(stats['save_path'], 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            dialogs = dataset.get('dialogs', [])
            if dialogs:
                print(f"\n📝 ПРИМЕР ДИАЛОГА:")
                sample = dialogs[0]
                print(f"   ID: {sample['id']}")
                print(f"   ПД: {'Да' if sample['has_pd'] else 'Нет'}")
                print(f"   Сущностей: {len(sample['entities'])}")
                print(f"   Текст: {sample['text'][:200]}...")
                
                if sample['entities']:
                    print(f"   Найденные сущности:")
                    for entity in sample['entities'][:3]:  # Первые 3
                        print(f"     - {entity['type']}: '{entity['text']}'")
                        
        except Exception as e:
            print(f"⚠️ Не удалось показать пример: {e}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Генерация прервана пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Ошибка генерации: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 