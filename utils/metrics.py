"""
Модуль для расчёта метрик качества извлечения сущностей (NER).

Основные принципы:
- Строгое совпадение (Exact Match) по умолчанию
- Эвристика: разрешаем смещение границ на 1 символ (|Δstart| + |Δend| ≤ 1)
- Опциональный relaxed режим с настраиваемым порогом IoU
- Только precision, recall, F1 (без accuracy)
"""

from typing import List, Dict, Tuple, Any
from collections import defaultdict


def calculate_overlap_score(entity1: Tuple[int, int, str], entity2: Tuple[int, int, str]) -> float:
    """Вычисляет IoU (Intersection over Union) для двух сущностей."""
    start1, end1, type1 = entity1
    start2, end2, type2 = entity2
    
    if type1 != type2:
        return 0.0
    
    # Находим пересечение
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start
    union_length = max(end1, end2) - min(start1, start2)
    
    return overlap_length / union_length if union_length > 0 else 0.0


def is_near_match(entity1: Tuple[int, int, str], entity2: Tuple[int, int, str]) -> bool:
    """
    Проверяет, являются ли две сущности 'почти совпадающими' (эвристика ±1 символ).
    Возвращает True, если |Δstart| + |Δend| ≤ 1 и типы совпадают.
    """
    start1, end1, type1 = entity1
    start2, end2, type2 = entity2
    
    if type1 != type2:
        return False
    
    start_diff = abs(start1 - start2)
    end_diff = abs(end1 - end2)
    
    return start_diff + end_diff <= 1


def match_spans(true_entities: List[Dict], predicted_entities: List[Dict], 
                overlap_threshold: float = 1.0) -> Tuple[int, int, int]:
    """
    Сопоставляет true и predicted сущности с учётом порога IoU.
    
    Args:
        true_entities: Список ground truth сущностей
        predicted_entities: Список предсказанных сущностей  
        overlap_threshold: Минимальный IoU для совпадения (1.0 = строгий режим)
    
    Returns:
        (tp, total_true, total_pred) - количество TP, общее кол-во true и pred
    """
    # Дедупликация: убираем точные дубли по координатам
    true_set = list({(e['start'], e['end'], e['type']) for e in true_entities})
    pred_set = list({(e['start'], e['end'], e['type']) for e in predicted_entities})
    
    matched_true = set()
    matched_pred = set()
    
    # 1. Ищем точные совпадения
    exact_matches = set(true_set) & set(pred_set)
    matched_true.update(exact_matches)
    matched_pred.update(exact_matches)
    
    # 2. Если строгий режим (IoU = 1.0), применяем эвристику ±1 символ
    if overlap_threshold >= 1.0:
        remaining_true = [e for e in true_set if e not in matched_true]
        remaining_pred = [e for e in pred_set if e not in matched_pred]
        
        for true_entity in remaining_true:
            best_match = None
            
            for pred_entity in remaining_pred:
                if pred_entity in matched_pred:
                    continue
                    
                if is_near_match(true_entity, pred_entity):
                    best_match = pred_entity
                    break  # Берём первое подходящее
            
            if best_match:
                matched_true.add(true_entity)
                matched_pred.add(best_match)
    
    # 3. Если не строгий режим, ищем перекрытия по IoU
    elif overlap_threshold > 0.0:
        remaining_true = [e for e in true_set if e not in matched_true]
        remaining_pred = [e for e in pred_set if e not in matched_pred]
        
        for true_entity in remaining_true:
            best_match = None
            best_score = 0.0
            
            for pred_entity in remaining_pred:
                if pred_entity in matched_pred:
                    continue
                    
                score = calculate_overlap_score(true_entity, pred_entity)
                if score > best_score and score >= overlap_threshold:
                    best_score = score
                    best_match = pred_entity
            
            if best_match:
                matched_true.add(true_entity)
                matched_pred.add(best_match)
    
    return len(matched_true), len(true_set), len(pred_set)


def calculate_metrics(true_entities: List[Dict], predicted_entities: List[Dict], 
                     overlap_threshold: float = 1.0) -> Dict[str, float]:
    """
    Вычисляет метрики precision, recall, F1 для извлечения сущностей.
    
    Args:
        true_entities: Ground truth сущности
        predicted_entities: Предсказанные сущности
        overlap_threshold: Порог IoU (1.0 = строгий режим)
    
    Returns:
        Словарь с метриками: precision, recall, f1, tp, fp, fn
    """
    tp, total_true, total_pred = match_spans(true_entities, predicted_entities, overlap_threshold)
    
    fp = total_pred - tp  # False Positives
    fn = total_true - tp  # False Negatives
    
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_true if total_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_metrics_by_type(true_entities: List[Dict], predicted_entities: List[Dict], 
                             overlap_threshold: float = 1.0) -> Dict[str, Dict[str, float]]:
    """
    Вычисляет метрики для каждого типа сущности отдельно.
    
    Args:
        true_entities: Ground truth сущности
        predicted_entities: Предсказанные сущности  
        overlap_threshold: Порог IoU (1.0 = строгий режим)
    
    Returns:
        Словарь {entity_type: {metric_name: value}}
        Включает ALL_MICRO (micro-averaged) и ALL_MACRO (macro-averaged) метрики
    """
    # Группируем сущности по типам
    true_by_type = defaultdict(list)
    pred_by_type = defaultdict(list)
    
    for entity in true_entities:
        true_by_type[entity['type']].append(entity)
    
    for entity in predicted_entities:
        pred_by_type[entity['type']].append(entity)
    
    # Получаем все уникальные типы
    all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
    
    # Общее количество истинных сущностей для расчёта долей
    total_true_entities = len({(e['start'], e['end'], e['type']) for e in true_entities})
    
    metrics_by_type = {}
    type_f1_scores = []  # Для macro averaging
    
    # Вычисляем метрики для каждого типа
    for entity_type in all_types:
        true_type_entities = true_by_type[entity_type]
        pred_type_entities = pred_by_type[entity_type]
        
        tp, total_true, total_pred = match_spans(true_type_entities, pred_type_entities, overlap_threshold)
        
        fp = total_pred - tp
        fn = total_true - tp
        
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Доля класса в датасете (по истинным сущностям)
        class_proportion = total_true / total_true_entities if total_true_entities > 0 else 0.0
        
        metrics_by_type[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_true': total_true,
            'total_pred': total_pred,
            'class_proportion': class_proportion
        }
        
        # Сохраняем F1 для macro averaging (только если есть истинные сущности этого типа)
        if total_true > 0:
            type_f1_scores.append(f1)
    
    # Добавляем общие метрики (ALL_MICRO - micro-averaged)
    overall_metrics = calculate_metrics(true_entities, predicted_entities, overlap_threshold)
    overall_metrics['total_true'] = len({(e['start'], e['end'], e['type']) for e in true_entities})
    overall_metrics['total_pred'] = len({(e['start'], e['end'], e['type']) for e in predicted_entities})
    overall_metrics['class_proportion'] = 1.0  # ALL включает все классы
    
    metrics_by_type['ALL_MICRO'] = overall_metrics
    
    # Добавляем макро-усреднённые метрики (ALL_MACRO)
    if type_f1_scores:
        macro_f1 = sum(type_f1_scores) / len(type_f1_scores)
        
        # Для macro precision и recall тоже считаем среднее по типам
        type_precisions = [metrics_by_type[t]['precision'] for t in all_types if metrics_by_type[t]['total_true'] > 0]
        type_recalls = [metrics_by_type[t]['recall'] for t in all_types if metrics_by_type[t]['total_true'] > 0]
        
        macro_precision = sum(type_precisions) / len(type_precisions) if type_precisions else 0.0
        macro_recall = sum(type_recalls) / len(type_recalls) if type_recalls else 0.0
    else:
        macro_f1 = 0.0
        macro_precision = 0.0
        macro_recall = 0.0
    
    # Суммарные TP, FP, FN для макро (информативно)
    total_tp = sum(metrics_by_type[t]['tp'] for t in all_types)
    total_fp = sum(metrics_by_type[t]['fp'] for t in all_types)
    total_fn = sum(metrics_by_type[t]['fn'] for t in all_types)
    
    metrics_by_type['ALL_MACRO'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_true': len({(e['start'], e['end'], e['type']) for e in true_entities}),
        'total_pred': len({(e['start'], e['end'], e['type']) for e in predicted_entities}),
        'class_proportion': 1.0  # ALL включает все классы
    }
    
    return metrics_by_type 