#!/usr/bin/env python3
"""
Анализатор качества аннотации: сравнивает исходный и аннотированный датасеты
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Загружает датасет, поддерживает оба формата."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    return data.get("dialogs", [])


def analyze_annotation_quality(original_path: str, annotated_path: str):
    """Анализирует качество аннотации."""
    original = load_dataset(original_path)
    annotated = load_dataset(annotated_path)
    
    # Создаем словарь для быстрого поиска
    original_dict = {dlg["id"]: dlg for dlg in original}
    
    stats = {
        "total_dialogs": len(annotated),
        "annotated_dialogs": 0,
        "text_changes": 0,
        "position_errors": 0,
        "has_pd_changes": {"false_to_true": 0, "true_to_false": 0},
        "entity_types": Counter(),
        "quality_issues": []
    }
    
    print("🔍 Анализ качества аннотации...")
    print("=" * 50)
    
    for ann_dlg in annotated:
        dlg_id = ann_dlg.get("id")
        orig_dlg = original_dict.get(dlg_id)
        
        if not orig_dlg:
            continue
            
        # Проверяем изменения текста
        if orig_dlg.get("text") != ann_dlg.get("text"):
            stats["text_changes"] += 1
            stats["quality_issues"].append(f"Диалог {dlg_id}: изменён текст")
            
        # Проверяем позиции entities
        text = ann_dlg.get("text", "")
        for entity in ann_dlg.get("entities", []):
            start, end = entity.get("start", 0), entity.get("end", 0)
            entity_text = entity.get("text", "")
            entity_type = entity.get("type", "")
            
            # Проверяем границы
            if start < 0 or end > len(text) or start >= end:
                stats["position_errors"] += 1
                stats["quality_issues"].append(f"Диалог {dlg_id}: некорректные позиции {start}-{end}")
                continue
                
            # Проверяем соответствие текста
            actual_text = text[start:end]
            if actual_text != entity_text:
                stats["position_errors"] += 1
                stats["quality_issues"].append(f"Диалог {dlg_id}: текст не совпадает '{entity_text}' != '{actual_text}'")
                continue
                
            # Считаем типы
            stats["entity_types"][entity_type] += 1
            
        # Проверяем изменения has_pd
        orig_has_pd = orig_dlg.get("has_pd", False)
        ann_has_pd = ann_dlg.get("has_pd", False)
        
        if orig_has_pd != ann_has_pd:
            if not orig_has_pd and ann_has_pd:
                stats["has_pd_changes"]["false_to_true"] += 1
            elif orig_has_pd and not ann_has_pd:
                stats["has_pd_changes"]["true_to_false"] += 1
                
        # Считаем аннотированные диалоги
        if ann_dlg.get("entities"):
            stats["annotated_dialogs"] += 1
    
    # Выводим результаты
    print(f"📊 Общая статистика:")
    print(f"  • Всего диалогов: {stats['total_dialogs']}")
    print(f"  • Аннотировано: {stats['annotated_dialogs']} ({stats['annotated_dialogs']/stats['total_dialogs']*100:.1f}%)")
    print()
    
    print(f"⚠️  Проблемы качества:")
    print(f"  • Изменения текста: {stats['text_changes']}")
    print(f"  • Ошибки позиций: {stats['position_errors']}")
    print()
    
    print(f"🔄 Изменения has_pd:")
    print(f"  • false → true: {stats['has_pd_changes']['false_to_true']}")
    print(f"  • true → false: {stats['has_pd_changes']['true_to_false']}")
    print()
    
    print(f"🏷️  Типы найденных сущностей:")
    for entity_type, count in stats["entity_types"].most_common():
        print(f"  • {entity_type}: {count}")
    print()
    
    # Оценка качества
    total_issues = stats["text_changes"] + stats["position_errors"]
    if total_issues == 0:
        quality = "ОТЛИЧНОЕ"
    elif total_issues < 5:
        quality = "ХОРОШЕЕ"
    elif total_issues < 20:
        quality = "СРЕДНЕЕ"
    else:
        quality = "ПЛОХОЕ"
        
    print(f"✅ Общая оценка качества: {quality}")
    
    # Показываем первые проблемы
    if stats["quality_issues"]:
        print(f"\n❌ Первые 10 проблем:")
        for issue in stats["quality_issues"][:10]:
            print(f"  • {issue}")
    
    return stats


def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Использование: python analyze_annotation_quality.py <исходный_датасет> <аннотированный_датасет>")
        sys.exit(1)
        
    original_path = sys.argv[1]
    annotated_path = sys.argv[2]
    
    if not Path(original_path).exists():
        print(f"❌ Файл не найден: {original_path}")
        sys.exit(1)
        
    if not Path(annotated_path).exists():
        print(f"❌ Файл не найден: {annotated_path}")
        sys.exit(1)
    
    analyze_annotation_quality(original_path, annotated_path)


if __name__ == "__main__":
    main() 