#!/usr/bin/env python3
"""
Скрипт для разбиения датасета на поддатасеты по заданному количеству диалогов
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any

def split_dataset(input_file: str, dialogs_per_subset: int = 50) -> None:
    """
    Разбивает датасет на поддатасеты
    
    Args:
        input_file: Путь к исходному файлу
        dialogs_per_subset: Количество диалогов в каждом поддатасете
    """
    # Читаем исходный файл
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_dialogs = len(data['dialogs'])
    print(f"Общее количество диалогов: {total_dialogs}")
    
    # Вычисляем количество поддатасетов
    num_subsets = (total_dialogs + dialogs_per_subset - 1) // dialogs_per_subset
    print(f"Будет создано поддатасетов: {num_subsets}")
    
    # Определяем базовое имя файла
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.dirname(input_file)
    
    # Создаем поддатасеты
    for i in range(num_subsets):
        start_idx = i * dialogs_per_subset
        end_idx = min((i + 1) * dialogs_per_subset, total_dialogs)
        
        # Берем диалоги для текущего поддатасета
        subset_dialogs = data['dialogs'][start_idx:end_idx]
        
        # Создаем новую структуру данных
        subset_data = {
            "metadata": {
                "total_dialogs": len(subset_dialogs),
                "generation_timestamp": datetime.now().timestamp(),
                "model_used": data['metadata'].get('model_used', 'unknown'),
                "parent_dataset": base_name,
                "subset_number": i + 1,
                "dialog_range": f"{start_idx + 1}-{end_idx}",
                "generation_stats": data['metadata'].get('generation_stats', {})
            },
            "dialogs": subset_dialogs
        }
        
        # Формируем имя файла с постфиксом
        output_file = os.path.join(output_dir, f"{base_name}.{i + 1}.json")
        
        # Сохраняем поддатасет
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, ensure_ascii=False, indent=2)
        
        print(f"Создан поддатасет {i + 1}: {output_file} ({len(subset_dialogs)} диалогов)")

def main():
    parser = argparse.ArgumentParser(description='Разбиение датасета на поддатасеты')
    parser.add_argument('input_file', help='Путь к исходному файлу датасета')
    parser.add_argument('--dialogs_per_subset', '-d', type=int, default=50,
                        help='Количество диалогов в каждом поддатасете (по умолчанию: 50)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Файл {args.input_file} не найден!")
        return 1
    
    try:
        split_dataset(args.input_file, args.dialogs_per_subset)
        print("Разбиение завершено!")
        return 0
    except Exception as e:
        print(f"Ошибка при разбиении: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 