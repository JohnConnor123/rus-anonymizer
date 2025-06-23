"""
Модуль для обработки и объединения датасетов с персональными данными.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import argparse
import glob
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Статистика датасета"""
    total_dialogs: int
    pd_dialogs: int
    non_pd_dialogs: int
    pd_percentage: float
    total_entities: int
    entity_types: Dict[str, int]


class DatasetProcessor:
    """Класс для обработки датасетов"""
    
    def __init__(self, input_files: List[str], output_dir: str = "data/processed"):
        self.input_files = input_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """Загружает датасет из файла"""
        logger.info(f"Загружаю датасет из {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"Загружено {len(data.get('dialogs', []))} диалогов")
        return data
    
    def calculate_stats(self, dialogs: List[Dict[str, Any]]) -> DatasetStats:
        """Вычисляет статистику датасета"""
        pd_count = sum(1 for d in dialogs if d.get('has_pd', False))
        total_entities = sum(len(d.get('entities', [])) for d in dialogs)
        
        # Подсчет типов сущностей
        entity_types = {}
        for dialog in dialogs:
            for entity in dialog.get('entities', []):
                entity_type = entity.get('type', 'UNKNOWN')
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        return DatasetStats(
            total_dialogs=len(dialogs),
            pd_dialogs=pd_count,
            non_pd_dialogs=len(dialogs) - pd_count,
            pd_percentage=pd_count / len(dialogs) * 100 if dialogs else 0,
            total_entities=total_entities,
            entity_types=entity_types
        )
    
    def merge_datasets(self) -> Dict[str, Any]:
        """Объединяет все датасеты"""
        logger.info("Начинаю объединение датасетов")
        
        all_dialogs = []
        all_metadata = []
        total_cost_tokens = 0
        
        for file_path in self.input_files:
            data = self.load_dataset(file_path)
            
            # Собираем диалоги
            dialogs = data.get('dialogs', [])
            all_dialogs.extend(dialogs)
            
            # Собираем метаданные
            metadata = data.get('metadata', {})
            all_metadata.append(metadata)
            
            # Суммируем токены
            gen_stats = metadata.get('generation_stats', {})
            total_cost_tokens += gen_stats.get('total_cost_tokens', 0)
        
        # Пересчитываем ID для уникальности
        for i, dialog in enumerate(all_dialogs, 1):
            dialog['id'] = i
        
        # Создаем объединенные метаданные
        stats = self.calculate_stats(all_dialogs)
        
        merged_metadata = {
            "total_dialogs": stats.total_dialogs,
            "merge_timestamp": datetime.now().timestamp(),
            "source_files": self.input_files,
            "merged_stats": {
                "total_dialogs": stats.total_dialogs,
                "pd_dialogs": stats.pd_dialogs,
                "non_pd_dialogs": stats.non_pd_dialogs,
                "pd_percentage": stats.pd_percentage,
                "total_entities": stats.total_entities,
                "entity_types": stats.entity_types,
                "total_cost_tokens": total_cost_tokens
            },
            "source_metadata": all_metadata
        }
        
        logger.info(f"Объединено {stats.total_dialogs} диалогов")
        logger.info(f"Диалогов с ПД: {stats.pd_dialogs} ({stats.pd_percentage:.1f}%)")
        
        return {
            "metadata": merged_metadata,
            "dialogs": all_dialogs
        }
    
    def split_dataset(self, data: Dict[str, Any], 
                     train_ratio: float = 0.7) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Разделяет датасет на train/val"""
        logger.info(f"Разделяю датасет в соотношении train={train_ratio:.1f}")
        
        dialogs = data['dialogs'].copy()
        random.shuffle(dialogs)
        
        split_idx = int(len(dialogs) * train_ratio)
        train_dialogs = dialogs[:split_idx]
        val_dialogs = dialogs[split_idx:]
        
        # Пересчитываем ID для каждого набора
        for i, dialog in enumerate(train_dialogs, 1):
            dialog['id'] = i
            
        for i, dialog in enumerate(val_dialogs, 1):
            dialog['id'] = i
        
        # Создаем метаданные для train
        train_stats = self.calculate_stats(train_dialogs)
        train_metadata = data['metadata'].copy()
        train_metadata.update({
            "split": "train",
            "split_timestamp": datetime.now().timestamp(),
            "split_ratio": train_ratio,
            "total_dialogs": train_stats.total_dialogs,
            "pd_dialogs": train_stats.pd_dialogs,
            "pd_percentage": train_stats.pd_percentage,
            "total_entities": train_stats.total_entities,
            "entity_types": train_stats.entity_types
        })
        
        # Создаем метаданные для val
        val_stats = self.calculate_stats(val_dialogs)
        val_metadata = data['metadata'].copy()
        val_metadata.update({
            "split": "val",
            "split_timestamp": datetime.now().timestamp(),
            "split_ratio": 1 - train_ratio,
            "total_dialogs": val_stats.total_dialogs,
            "pd_dialogs": val_stats.pd_dialogs,
            "pd_percentage": val_stats.pd_percentage,
            "total_entities": val_stats.total_entities,
            "entity_types": val_stats.entity_types
        })
        
        train_data = {
            "metadata": train_metadata,
            "dialogs": train_dialogs
        }
        
        val_data = {
            "metadata": val_metadata,
            "dialogs": val_dialogs
        }
        
        logger.info(f"Train: {len(train_dialogs)} диалогов")
        logger.info(f"Val: {len(val_dialogs)} диалогов")
        
        return train_data, val_data
    
    def save_dataset(self, data: Dict[str, Any], filename: str):
        """Сохраняет датасет в файл"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Датасет сохранен: {output_path}")
    
    def process_all(self, train_ratio: float = 0.7, seed: int = 42):
        """Полная обработка: объединение и разделение"""
        random.seed(seed)
        
        # Объединяем датасеты
        merged_data = self.merge_datasets()
        
        # Сохраняем объединенный датасет
        self.save_dataset(merged_data, "merged_dataset.json")
        
        # Разделяем на train/val
        train_data, val_data = self.split_dataset(merged_data, train_ratio)
        
        # Сохраняем разделенные датасеты
        self.save_dataset(train_data, "train_dataset.json")
        self.save_dataset(val_data, "val_dataset.json")
        
        return {
            "merged": merged_data,
            "train": train_data,
            "val": val_data
        }


def analyze_dataset(file_path: str):
    """Анализирует один датасет и выводит статистику"""
    processor = DatasetProcessor([file_path])
    data = processor.load_dataset(file_path)
    stats = processor.calculate_stats(data.get('dialogs', []))
    
    print(f"\n📊 Статистика датасета: {file_path}")
    print(f"Всего диалогов: {stats.total_dialogs}")
    print(f"Диалогов с ПД: {stats.pd_dialogs} ({stats.pd_percentage:.1f}%)")
    print(f"Диалогов без ПД: {stats.non_pd_dialogs}")
    print(f"Всего сущностей: {stats.total_entities}")
    
    if stats.entity_types:
        print("\nТипы сущностей:")
        for entity_type, count in sorted(stats.entity_types.items()):
            print(f"  {entity_type}: {count}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Обработка датасетов")
    parser.add_argument('--input', help='Входной файл датасета')
    parser.add_argument('--merge', nargs='+', help='Файлы для объединения (можно указать несколько или использовать glob)')
    parser.add_argument('--output', help='Выходной файл')
    parser.add_argument('--analyze-only', action='store_true', help='Только анализ без сохранения')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Доля train датасета')
    
    args = parser.parse_args()
    
    if args.analyze_only and args.input:
        analyze_dataset(args.input)
        return
        
    if args.merge:
        # Если передан паттерн glob, расширяем его
        files = []
        for pattern in args.merge:
            if '*' in pattern or '?' in pattern:
                files.extend(glob.glob(pattern))
            else:
                files.append(pattern)
        
        if not files:
            print(f"❌ Не найдено файлов для объединения")
            return
            
        processor = DatasetProcessor(files)
        merged_data = processor.merge_datasets()
        
        if args.output:
            processor.save_dataset(merged_data, Path(args.output).name)
            print(f"✅ Датасеты объединены и сохранены в: {args.output}")
        else:
            print("✅ Датасеты объединены (результат не сохранен)")
            
    elif args.input:
        processor = DatasetProcessor([args.input])
        # Дополнительная обработка для одного файла
        print(f"📁 Обрабатываю файл: {args.input}")
        analyze_dataset(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 