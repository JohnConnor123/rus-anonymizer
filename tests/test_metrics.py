#!/usr/bin/env python3
"""
Тесты для модуля метрик (utils/metrics.py).
"""

import pytest
import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics import (
    calculate_overlap_score,
    is_near_match,
    match_spans,
    calculate_metrics,
    calculate_metrics_by_type
)


class TestOverlapScore:
    """Тесты для функции calculate_overlap_score."""
    
    def test_exact_match(self):
        """Тест точного совпадения."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (10, 20, 'PERSON')
        assert calculate_overlap_score(entity1, entity2) == 1.0
    
    def test_no_overlap(self):
        """Тест отсутствия перекрытия."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (25, 30, 'PERSON')
        assert calculate_overlap_score(entity1, entity2) == 0.0
    
    def test_different_types(self):
        """Тест разных типов сущностей."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (10, 20, 'AGE')
        assert calculate_overlap_score(entity1, entity2) == 0.0
    
    def test_partial_overlap(self):
        """Тест частичного перекрытия."""
        entity1 = (10, 20, 'PERSON')  # длина 10
        entity2 = (15, 25, 'PERSON')  # длина 10, пересечение 5
        # IoU = 5 / (10 + 10 - 5) = 5/15 = 1/3
        expected = 5.0 / 15.0
        assert abs(calculate_overlap_score(entity1, entity2) - expected) < 1e-6


class TestNearMatch:
    """Тесты для функции is_near_match."""
    
    def test_exact_match(self):
        """Тест точного совпадения."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (10, 20, 'PERSON')
        assert is_near_match(entity1, entity2) == True
    
    def test_one_symbol_start_diff(self):
        """Тест смещения начала на 1 символ."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (11, 20, 'PERSON')
        assert is_near_match(entity1, entity2) == True
    
    def test_one_symbol_end_diff(self):
        """Тест смещения конца на 1 символ."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (10, 21, 'PERSON')
        assert is_near_match(entity1, entity2) == True
    
    def test_two_symbols_diff(self):
        """Тест смещения на 2 символа (не должно совпадать)."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (11, 21, 'PERSON')
        # |11-10| + |21-20| = 1 + 1 = 2 > 1
        assert is_near_match(entity1, entity2) == False
    
    def test_different_types(self):
        """Тест разных типов."""
        entity1 = (10, 20, 'PERSON')
        entity2 = (10, 20, 'AGE')
        assert is_near_match(entity1, entity2) == False


class TestMatchSpans:
    """Тесты для функции match_spans."""
    
    def test_strict_exact_match(self):
        """Тест строгого точного совпадения."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 30, 'end': 35, 'type': 'AGE'}
        ]
        pred_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 30, 'end': 35, 'type': 'AGE'}
        ]
        
        tp, total_true, total_pred = match_spans(true_entities, pred_entities, overlap_threshold=1.0)
        assert tp == 2
        assert total_true == 2
        assert total_pred == 2
    
    def test_strict_near_match(self):
        """Тест строгого режима с эвристикой ±1 символ."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'}
        ]
        pred_entities = [
            {'start': 11, 'end': 20, 'type': 'PERSON'}  # смещение на 1 символ
        ]
        
        tp, total_true, total_pred = match_spans(true_entities, pred_entities, overlap_threshold=1.0)
        assert tp == 1
        assert total_true == 1
        assert total_pred == 1
    
    def test_strict_no_match(self):
        """Тест строгого режима без совпадений."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'}
        ]
        pred_entities = [
            {'start': 12, 'end': 22, 'type': 'PERSON'}  # смещение на 2 символа
        ]
        
        tp, total_true, total_pred = match_spans(true_entities, pred_entities, overlap_threshold=1.0)
        assert tp == 0
        assert total_true == 1
        assert total_pred == 1
    
    def test_relaxed_partial_match(self):
        """Тест relaxed режима с частичным перекрытием."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'}
        ]
        pred_entities = [
            {'start': 15, 'end': 25, 'type': 'PERSON'}  # IoU = 5/15 = 0.33
        ]
        
        tp, total_true, total_pred = match_spans(true_entities, pred_entities, overlap_threshold=0.3)
        assert tp == 1
        assert total_true == 1
        assert total_pred == 1
    
    def test_deduplication(self):
        """Тест дедупликации предсказаний."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'}
        ]
        pred_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 10, 'end': 20, 'type': 'PERSON'}  # дубликат
        ]
        
        tp, total_true, total_pred = match_spans(true_entities, pred_entities, overlap_threshold=1.0)
        assert tp == 1
        assert total_true == 1
        assert total_pred == 1  # дубликат удалён


class TestCalculateMetrics:
    """Тесты для функции calculate_metrics."""
    
    def test_perfect_match(self):
        """Тест идеального совпадения."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 30, 'end': 35, 'type': 'AGE'}
        ]
        pred_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 30, 'end': 35, 'type': 'AGE'}
        ]
        
        metrics = calculate_metrics(true_entities, pred_entities)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['tp'] == 2
        assert metrics['fp'] == 0
        assert metrics['fn'] == 0
    
    def test_no_predictions(self):
        """Тест отсутствия предсказаний."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'}
        ]
        pred_entities = []
        
        metrics = calculate_metrics(true_entities, pred_entities)
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['tp'] == 0
        assert metrics['fp'] == 0
        assert metrics['fn'] == 1
    
    def test_false_positives(self):
        """Тест ложных срабатываний."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'}
        ]
        pred_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 30, 'end': 35, 'type': 'AGE'}  # ложное срабатывание
        ]
        
        metrics = calculate_metrics(true_entities, pred_entities)
        assert metrics['precision'] == 0.5  # 1/2
        assert metrics['recall'] == 1.0     # 1/1
        assert metrics['f1'] == 2 * 0.5 * 1.0 / (0.5 + 1.0)  # 2/3
        assert metrics['tp'] == 1
        assert metrics['fp'] == 1
        assert metrics['fn'] == 0


class TestCalculateMetricsByType:
    """Тесты для функции calculate_metrics_by_type."""
    
    def test_metrics_by_type(self):
        """Тест метрик по типам сущностей."""
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 30, 'end': 35, 'type': 'AGE'},
            {'start': 40, 'end': 50, 'type': 'PERSON'}
        ]
        pred_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},  # TP для PERSON
            {'start': 32, 'end': 37, 'type': 'AGE'},     # FP для AGE (не совпадает)
            {'start': 60, 'end': 70, 'type': 'LOCATION'} # FP для LOCATION
        ]
        
        metrics_by_type = calculate_metrics_by_type(true_entities, pred_entities)
        
        # Проверяем PERSON
        person_metrics = metrics_by_type['PERSON']
        assert person_metrics['tp'] == 1
        assert person_metrics['fp'] == 0
        assert person_metrics['fn'] == 1
        assert person_metrics['precision'] == 1.0
        assert person_metrics['recall'] == 0.5
        assert person_metrics['class_proportion'] == 2.0/3.0  # 2 из 3 истинных сущностей
        
        # Проверяем AGE
        age_metrics = metrics_by_type['AGE']
        assert age_metrics['tp'] == 0
        assert age_metrics['fp'] == 1
        assert age_metrics['fn'] == 1
        assert age_metrics['precision'] == 0.0
        assert age_metrics['recall'] == 0.0
        assert age_metrics['class_proportion'] == 1.0/3.0  # 1 из 3 истинных сущностей
        
        # Проверяем LOCATION
        location_metrics = metrics_by_type['LOCATION']
        assert location_metrics['tp'] == 0
        assert location_metrics['fp'] == 1
        assert location_metrics['fn'] == 0
        assert location_metrics['precision'] == 0.0
        assert location_metrics['recall'] == 0.0
        assert location_metrics['class_proportion'] == 0.0  # 0 из 3 истинных сущностей
        
        # Проверяем общие метрики ALL_MICRO
        micro_metrics = metrics_by_type['ALL_MICRO']
        assert micro_metrics['tp'] == 1
        assert micro_metrics['fp'] == 2
        assert micro_metrics['fn'] == 2
        assert micro_metrics['class_proportion'] == 1.0
        
        # Проверяем макро метрики ALL_MACRO
        macro_metrics = metrics_by_type['ALL_MACRO']
        # Macro F1 = среднее F1 по типам с истинными сущностями
        # PERSON F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 2/3
        # AGE F1 = 0.0
        # Macro F1 = (2/3 + 0.0) / 2 = 1/3
        expected_macro_f1 = (2.0/3.0 + 0.0) / 2.0
        assert abs(macro_metrics['f1'] - expected_macro_f1) < 1e-6
        assert macro_metrics['class_proportion'] == 1.0
    
    def test_no_accuracy_in_metrics(self):
        """Тест отсутствия accuracy в метриках."""
        true_entities = [{'start': 10, 'end': 20, 'type': 'PERSON'}]
        pred_entities = [{'start': 10, 'end': 20, 'type': 'PERSON'}]
        
        metrics_by_type = calculate_metrics_by_type(true_entities, pred_entities)
        
        # Проверяем, что accuracy отсутствует
        for entity_type, metrics in metrics_by_type.items():
            assert 'accuracy' not in metrics
            # Проверяем наличие нужных метрик
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'tp' in metrics
            assert 'fp' in metrics
            assert 'fn' in metrics
            assert 'class_proportion' in metrics
    
    def test_micro_vs_macro_difference(self):
        """Тест различия между micro и macro метриками."""
        # Создаём несбалансированный датасет где один класс доминирует
        true_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},
            {'start': 30, 'end': 40, 'type': 'PERSON'},
            {'start': 50, 'end': 60, 'type': 'PERSON'},
            {'start': 70, 'end': 80, 'type': 'PERSON'},
            {'start': 90, 'end': 95, 'type': 'AGE'}  # Только одна AGE сущность
        ]
        pred_entities = [
            {'start': 10, 'end': 20, 'type': 'PERSON'},  # TP
            {'start': 30, 'end': 40, 'type': 'PERSON'},  # TP
            # Пропускаем остальные PERSON (2 FN)
            # Пропускаем AGE (1 FN)
        ]
        
        metrics_by_type = calculate_metrics_by_type(true_entities, pred_entities)
        
        micro_metrics = metrics_by_type['ALL_MICRO']
        macro_metrics = metrics_by_type['ALL_MACRO']
        
        # Micro: TP=2, FP=0, FN=3
        # Micro Precision = 2/2 = 1.0, Micro Recall = 2/5 = 0.4
        assert micro_metrics['precision'] == 1.0
        assert micro_metrics['recall'] == 0.4
        
        # Macro: среднее по классам
        # PERSON: P=1.0, R=0.5, F1=2/3
        # AGE: P=0.0, R=0.0, F1=0.0
        # Macro F1 = (2/3 + 0.0) / 2 = 1/3
        expected_macro_f1 = (2.0/3.0) / 2.0
        assert abs(macro_metrics['f1'] - expected_macro_f1) < 1e-6
        
        # Micro и Macro должны различаться
        assert micro_metrics['f1'] != macro_metrics['f1']


if __name__ == "__main__":
    pytest.main([__file__]) 