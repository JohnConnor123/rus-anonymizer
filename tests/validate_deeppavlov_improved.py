#!/usr/bin/env python3
"""
Улучшенная валидация DeepPavlov моделей.
Использует более умную логику сравнения сущностей.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from anonymizers.deeppavlov_ner.deeppavlov_anonymizer import DeepPavlovAnonymizer
    DEEPPAVLOV_AVAILABLE = True
except ImportError as e:
    print(f"❌ DeepPavlov не доступен: {e}")
    DEEPPAVLOV_AVAILABLE = False


def load_test_dataset(filename: str = "test_dataset.json") -> List[Dict]:
    """Загружает тестовый датасет."""
    test_file = project_root / "tests" / filename
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {test_file} не найден")
        return []


def entities_overlap(entity1: Tuple[int, int, str], entity2: Tuple[int, int, str]) -> bool:
    """Проверяет, перекрываются ли две сущности."""
    start1, end1, type1 = entity1
    start2, end2, type2 = entity2
    
    # Типы должны совпадать
    if type1 != type2:
        return False
    
    # Проверяем перекрытие координат
    return not (end1 <= start2 or end2 <= start1)


def calculate_overlap_score(entity1: Tuple[int, int, str], entity2: Tuple[int, int, str]) -> float:
    """Вычисляет степень перекрытия двух сущностей (0.0 - 1.0)."""
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


def merge_person_entities(entities: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
    """Объединяет соседние сущности PERSON в одну."""
    if not entities:
        return entities
    
    # Сортируем по позиции
    sorted_entities = sorted(entities, key=lambda x: x[0])
    merged = []
    
    i = 0
    while i < len(sorted_entities):
        start, end, entity_type, value = sorted_entities[i]
        
        if entity_type == 'PERSON':
            # Ищем соседние PERSON сущности
            merged_start = start
            merged_end = end
            merged_value = value
            
            j = i + 1
            while j < len(sorted_entities):
                next_start, next_end, next_type, next_value = sorted_entities[j]
                
                # Если следующая сущность тоже PERSON и находится рядом (с учетом пробелов)
                if (next_type == 'PERSON' and 
                    next_start <= merged_end + 3):  # +3 для учета пробелов
                    merged_end = next_end
                    merged_value = merged_value + " " + next_value if not merged_value.endswith(next_value) else merged_value
                    j += 1
                else:
                    break
            
            merged.append((merged_start, merged_end, entity_type, merged_value.strip()))
            i = j
        else:
            merged.append((start, end, entity_type, value))
            i += 1
    
    return merged


def improve_age_entities(entities: List[Tuple[int, int, str, str]], text: str) -> List[Tuple[int, int, str, str]]:
    """Улучшает извлечение возраста - убирает лишние слова."""
    improved = []
    
    for start, end, entity_type, value in entities:
        if entity_type == 'AGE':
            # Ищем числовую часть в тексте
            import re
            age_match = re.search(r'\d+\s*лет', value)
            if age_match:
                # Находим позицию числовой части в оригинальном тексте
                age_text = age_match.group()
                age_start = text.find(age_text, start)
                if age_start != -1:
                    age_end = age_start + len(age_text)
                    improved.append((age_start, age_end, entity_type, age_text))
                    continue
        
        improved.append((start, end, entity_type, value))
    
    return improved


def smart_entity_matching(true_entities: List[Dict], found_entities: List[Tuple[int, int, str, str]], 
                         text: str, overlap_threshold: float = 0.3) -> Tuple[int, int, int]:
    """
    Умное сопоставление сущностей с учетом перекрытий.
    Возвращает (правильные, всего_истинных, всего_найденных).
    """
    # Улучшаем найденные сущности
    found_entities = merge_person_entities(found_entities)
    found_entities = improve_age_entities(found_entities, text)
    
    # Преобразуем в удобный формат
    true_set = [(e['start'], e['end'], e['type']) for e in true_entities]
    found_set = [(start, end, entity_type) for start, end, entity_type, _ in found_entities]
    
    matched_true = set()
    matched_found = set()
    
    # Ищем точные совпадения
    exact_matches = set(true_set) & set(found_set)
    matched_true.update(exact_matches)
    matched_found.update(exact_matches)
    
    # Ищем перекрытия для несовпавших
    remaining_true = [e for e in true_set if e not in matched_true]
    remaining_found = [e for e in found_set if e not in matched_found]
    
    for true_entity in remaining_true:
        best_match = None
        best_score = 0.0
        
        for found_entity in remaining_found:
            if found_entity in matched_found:
                continue
                
            score = calculate_overlap_score(true_entity, found_entity)
            if score > best_score and score >= overlap_threshold:
                best_score = score
                best_match = found_entity
        
        if best_match:
            matched_true.add(true_entity)
            matched_found.add(best_match)
    
    return len(matched_true), len(true_entities), len(found_entities)


def analyze_deeppavlov_improved():
    """Улучшенный анализ работы DeepPavlov."""
    if not DEEPPAVLOV_AVAILABLE:
        print("❌ DeepPavlov недоступен для анализа")
        return
    
    print("🧠 Улучшенная валидация DeepPavlov NER")
    print("=" * 60)
    
    # Загружаем тестовые данные
    test_data = load_test_dataset()
    if not test_data:
        print("❌ Нет тестовых данных")
        return
    
    print(f"📋 Загружено {len(test_data)} примеров")
    
    # Инициализируем анонимизатор
    print("\n🔧 Инициализация DeepPavlov...")
    start_init = time.time()
    
    try:
        anonymizer = DeepPavlovAnonymizer(aggressiveness=0.8)
        init_time = time.time() - start_init
        print(f"✅ Инициализация завершена за {init_time:.2f}с")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # Анализируем каждый пример
    total_correct = 0
    total_true = 0
    total_found = 0
    
    for i, example in enumerate(test_data, 1):
        print(f"\n📝 Пример {i}/{len(test_data)}")
        print(f"Текст: {example['text']}")
        
        # Ожидаемые сущности
        true_entities = example['entities']
        
        # Извлекаем сущности
        start_time = time.time()
        try:
            extracted = anonymizer.extract_entities(example['text'])
            process_time = time.time() - start_time
            
            # Умное сопоставление
            correct, true_count, found_count = smart_entity_matching(
                true_entities, extracted, example['text']
            )
            
            total_correct += correct
            total_true += true_count
            total_found += found_count
            
            print(f"Ожидаемые сущности ({true_count}):")
            for entity in true_entities:
                text_part = example['text'][entity['start']:entity['end']]
                print(f"  - {entity['type']}: '{text_part}' [{entity['start']}:{entity['end']}]")
            
            print(f"Найденные сущности ({found_count}) за {process_time*1000:.1f}мс:")
            # Показываем улучшенные сущности
            improved_entities = merge_person_entities(extracted)
            improved_entities = improve_age_entities(improved_entities, example['text'])
            
            for start, end, entity_type, value in improved_entities:
                print(f"  - {entity_type}: '{value}' [{start}:{end}]")
            
            accuracy = correct / true_count if true_count > 0 else 0
            print(f"✅ Совпадений: {correct}/{true_count} (точность: {accuracy*100:.1f}%)")
                    
        except Exception as e:
            print(f"❌ Ошибка обработки: {e}")
            continue
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("📊 УЛУЧШЕННАЯ СТАТИСТИКА")
    print("=" * 60)
    
    precision = total_correct / total_found if total_found > 0 else 0
    recall = total_correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Всего ожидаемых сущностей: {total_true}")
    print(f"Всего найденных сущностей: {total_found}")
    print(f"Правильно найденных: {total_correct}")
    print(f"")
    print(f"🎯 Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"🎯 Recall: {recall:.3f} ({recall*100:.1f}%)")
    print(f"🎯 F1-Score: {f1:.3f} ({f1*100:.1f}%)")
    
    # Анализ производительности
    print(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ")
    print(f"Время инициализации: {init_time:.2f}с")
    print(f"Среднее время на пример: {(time.time() - start_init - init_time) / len(test_data) * 1000:.1f}мс")


def main():
    """Основная функция."""
    print("🚀 Улучшенная валидация DeepPavlov моделей")
    print("=" * 50)
    
    analyze_deeppavlov_improved()


if __name__ == "__main__":
    main() 