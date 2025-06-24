"""
Утилиты для работы с метаданными датасетов.
"""

from typing import Dict, Any, List
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class MetadataValidator:
    """Валидатор метаданных"""
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """Проверяет корректность метаданных"""
        required_fields = ['total_dialogs']
        
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Отсутствует обязательное поле: {field}")
                return False
        
        return True
    
    @staticmethod
    def validate_dialog(dialog: Dict[str, Any]) -> bool:
        """Проверяет корректность диалога"""
        required_fields = ['text', 'entities', 'has_pd', 'id']
        
        for field in required_fields:
            if field not in dialog:
                logger.warning(f"Диалог {dialog.get('id', '?')}: отсутствует поле {field}")
                return False
        
        # Проверяем entities
        if not isinstance(dialog['entities'], list):
            logger.warning(f"Диалог {dialog['id']}: entities должен быть списком")
            return False
            
        for entity in dialog['entities']:
            if not all(key in entity for key in ['start', 'end', 'type', 'text']):
                logger.warning(f"Диалог {dialog['id']}: некорректная сущность")
                return False
        
        return True


class MetadataUpdater:
    """Класс для обновления метаданных"""
    
    @staticmethod
    def update_generation_stats(metadata: Dict[str, Any], 
                              new_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Обновляет статистику генерации"""
        if 'generation_stats' not in metadata:
            metadata['generation_stats'] = {}
        
        metadata['generation_stats'].update(new_stats)
        metadata['last_updated'] = datetime.now().timestamp()
        
        return metadata
    
    @staticmethod
    def add_annotation_info(metadata: Dict[str, Any], 
                           annotation_type: str, 
                           count: int) -> Dict[str, Any]:
        """Добавляет информацию об аннотации"""
        annotation_key = f"{annotation_type}_annotation"
        metadata[annotation_key] = count
        metadata[f"{annotation_key}_timestamp"] = datetime.now().timestamp()
        
        return metadata
    
    @staticmethod
    def merge_metadata_list(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединяет список метаданных"""
        if not metadata_list:
            return {}
        
        merged = {
            "merge_timestamp": datetime.now().timestamp(),
            "source_count": len(metadata_list),
            "total_dialogs": 0,
            "total_cost_tokens": 0,
            "models_used": set(),
            "generation_configs": []
        }
        
        for metadata in metadata_list:
            # Суммируем диалоги
            merged["total_dialogs"] += metadata.get("total_dialogs", 0)
            
            # Суммируем токены
            gen_stats = metadata.get("generation_stats", {})
            merged["total_cost_tokens"] += gen_stats.get("total_cost_tokens", 0)
            
            # Собираем модели
            model_used = metadata.get("model_used")
            if model_used:
                merged["models_used"].add(model_used)
            
            # Собираем конфигурации генерации
            configs = gen_stats.get("generation_configs", [])
            merged["generation_configs"].extend(configs)
        
        # Конвертируем set в list для JSON
        merged["models_used"] = list(merged["models_used"])
        
        return merged


class MetadataAnalyzer:
    """Анализатор метаданных"""
    
    @staticmethod
    def analyze_dataset(data: Dict[str, Any]) -> Dict[str, Any]:
        """Проводит анализ датасета"""
        dialogs = data.get('dialogs', [])
        metadata = data.get('metadata', {})
        
        # Базовая статистика
        pd_count = sum(1 for d in dialogs if d.get('has_pd', False))
        total_entities = sum(len(d.get('entities', [])) for d in dialogs)
        
        # Анализ типов сущностей
        entity_types = {}
        entity_lengths = []
        
        for dialog in dialogs:
            entities = dialog.get('entities', [])
            entity_lengths.append(len(entities))
            
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Анализ длин текстов
        text_lengths = [len(d.get('text', '')) for d in dialogs]
        
        analysis = {
            "total_dialogs": len(dialogs),
            "pd_dialogs": pd_count,
            "pd_percentage": pd_count / len(dialogs) * 100 if dialogs else 0,
            "total_entities": total_entities,
            "avg_entities_per_dialog": total_entities / len(dialogs) if dialogs else 0,
            "entity_types": entity_types,
            "text_stats": {
                "avg_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                "min_length": min(text_lengths) if text_lengths else 0,
                "max_length": max(text_lengths) if text_lengths else 0
            },
            "entity_stats": {
                "avg_per_dialog": sum(entity_lengths) / len(entity_lengths) if entity_lengths else 0,
                "max_per_dialog": max(entity_lengths) if entity_lengths else 0
            }
        }
        
        return analysis
    
    @staticmethod
    def compare_datasets(data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
        """Сравнивает два датасета"""
        analysis1 = MetadataAnalyzer.analyze_dataset(data1)
        analysis2 = MetadataAnalyzer.analyze_dataset(data2)
        
        comparison = {
            "dataset1": analysis1,
            "dataset2": analysis2,
            "differences": {
                "dialog_count_diff": analysis2["total_dialogs"] - analysis1["total_dialogs"],
                "pd_percentage_diff": analysis2["pd_percentage"] - analysis1["pd_percentage"],
                "entity_count_diff": analysis2["total_entities"] - analysis1["total_entities"]
            }
        }
        
        return comparison
    
    @staticmethod
    def generate_report(data: Dict[str, Any]) -> str:
        """Генерирует текстовый отчет по датасету"""
        analysis = MetadataAnalyzer.analyze_dataset(data)
        
        report = f"""
=== ОТЧЕТ ПО ДАТАСЕТУ ===

Общая информация:
- Всего диалогов: {analysis['total_dialogs']}
- Диалогов с ПД: {analysis['pd_dialogs']} ({analysis['pd_percentage']:.1f}%)
- Всего сущностей: {analysis['total_entities']}
- Среднее количество сущностей на диалог: {analysis['avg_entities_per_dialog']:.1f}

Статистика по тексту:
- Средняя длина: {analysis['text_stats']['avg_length']:.0f} символов
- Минимальная длина: {analysis['text_stats']['min_length']} символов
- Максимальная длина: {analysis['text_stats']['max_length']} символов

Типы сущностей:
"""
        
        for entity_type, count in sorted(analysis['entity_types'].items()):
            percentage = count / analysis['total_entities'] * 100
            report += f"- {entity_type}: {count} ({percentage:.1f}%)\n"
        
        return report


def save_metadata_report(data: Dict[str, Any], output_path: str):
    """Сохраняет отчет по метаданным в файл"""
    report = MetadataAnalyzer.generate_report(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Отчет сохранен: {output_path}") 