#!/usr/bin/env python3
"""
Улучшенная валидация DeepPavlov моделей.
Использует более умную логику сравнения сущностей.
"""

import json
import time
import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импорт обновлённого модуля метрик
from utils.metrics import calculate_metrics, calculate_metrics_by_type

try:
    from anonymizers.deeppavlov_ner.deeppavlov_anonymizer import DeepPavlovAnonymizer
    DEEPPAVLOV_AVAILABLE = True
except ImportError as e:
    print(f"❌ DeepPavlov не доступен: {e}")
    DEEPPAVLOV_AVAILABLE = False


def load_test_dataset(filename: str = "test_dataset.json") -> List[Dict]:
    """Загружает тестовый датасет."""
    test_file = project_root / "data" / filename
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {test_file} не найден")
        return []


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


def get_dataset_name(dataset_path: str) -> str:
    """Извлекает название датасета из пути к файлу."""
    return Path(dataset_path).stem


def save_metrics_to_csv(metrics_by_type: Dict[str, Dict[str, float]], 
                       method_name: str, dataset_name: str, output_dir: Path):
    """Сохраняет метрики анонимизатора в CSV файл."""
    # Создаем безопасное имя файла
    safe_method_name = method_name.replace(" ", "_").replace("/", "_")
    filename = f"{dataset_name}_{safe_method_name}_metrics.csv"
    filepath = output_dir / filename
    
    # Создаем директорию если не существует
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['entity_type', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'total_true', 'total_pred', 'class_proportion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entity_type, metrics in metrics_by_type.items():
            row = {'entity_type': entity_type}
            row.update(metrics)
            writer.writerow(row)
    
    print(f"💾 Метрики {method_name} сохранены в {filepath}")


def update_combined_metrics_csv(metrics_by_type: Dict[str, Dict[str, float]], 
                               method_name: str, dataset_name: str, output_dir: Path) -> bool:
    """
    Обновляет существующую сводную таблицу, добавляя данные DeepPavlov если их там нет.
    Возвращает True если обновление выполнено, False если файл не найден или данные уже есть.
    """
    combined_filename = f"{dataset_name}_combined_metrics.csv"
    combined_filepath = output_dir / combined_filename
    
    # Проверяем существование сводной таблицы
    if not combined_filepath.exists():
        print(f"⚠️  Сводная таблица {combined_filepath} не найдена")
        return False
    
    # Читаем существующие данные
    existing_data = []
    with open(combined_filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            existing_data.append(row)
    
    # Проверяем, есть ли уже данные для нашего метода
    existing_methods = {row['method'] for row in existing_data}
    if method_name in existing_methods:
        print(f"⚠️  Данные для {method_name} уже есть в сводной таблице")
        return False
    
    # Добавляем наши данные
    new_rows = []
    for entity_type, metrics in metrics_by_type.items():
        row = {
            'method': method_name,
            'entity_type': entity_type
        }
        row.update(metrics)
        new_rows.append(row)
    
    # Записываем обновленные данные
    all_data = existing_data + new_rows
    with open(combined_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['method', 'entity_type', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'total_true', 'total_pred', 'class_proportion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
    
    print(f"✅ Данные {method_name} добавлены в сводную таблицу {combined_filepath}")
    return True


def validate_deeppavlov(test_data: List[Dict], dataset_name: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Валидирует DeepPavlov анонимизатор и возвращает общие метрики и метрики по типам."""
    print("\n🔍 Валидация DeepPavlov...")
    
    start_time = time.time()
    
    # Инициализируем анонимизатор
    print("🔧 Инициализация DeepPavlov...")
    start_init = time.time()
    
    try:
        anonymizer = DeepPavlovAnonymizer(aggressiveness=0.8)
        init_time = time.time() - start_init
        print(f"✅ Инициализация завершена за {init_time:.2f}с")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        raise
    
    all_true_entities = []
    all_predicted_entities = []
    processing_times = []
    
    for i, example in enumerate(test_data):
        text = example['text']
        true_entities = example['entities']
        
        # Измеряем время обработки
        process_start = time.time()
        
        try:
            # Извлекаем сущности
            extracted = anonymizer.extract_entities(text)
            
            # Улучшаем извлеченные сущности
            improved_entities = merge_person_entities(extracted)
            improved_entities = improve_age_entities(improved_entities, text)
            
            # Преобразуем в формат для сравнения
            predicted_entities = []
            for start, end, entity_type, value in improved_entities:
                predicted_entities.append({
                    'start': start,
                    'end': end,
                    'type': entity_type,
                    'value': value
                })
            
            all_true_entities.extend(true_entities)
            all_predicted_entities.extend(predicted_entities)
            
        except Exception as e:
            print(f"❌ Ошибка в примере {i+1}: {e}")
            continue
        
        process_time = time.time() - process_start
        processing_times.append(process_time)
    
    total_time = time.time() - start_time
    
    # Вычисляем общие метрики (строгий режим по умолчанию)
    overall_metrics = calculate_metrics(all_true_entities, all_predicted_entities)
    
    # Вычисляем метрики по типам (строгий режим по умолчанию)
    metrics_by_type = calculate_metrics_by_type(all_true_entities, all_predicted_entities)
    
    # Добавляем информацию о производительности к общим метрикам
    overall_metrics.update({
        'total_time': total_time,
        'init_time': init_time,
        'avg_time_per_example': sum(processing_times) / len(processing_times) if processing_times else 0,
        'examples_processed': len(processing_times)
    })
    
    return overall_metrics, metrics_by_type


def analyze_deeppavlov_improved(dataset_path: str = None, update_pivot: bool = False):
    """Улучшенный анализ работы DeepPavlov."""
    if not DEEPPAVLOV_AVAILABLE:
        print("❌ DeepPavlov недоступен для анализа")
        return
    
    print("🧠 Улучшенная валидация DeepPavlov NER")
    print("=" * 60)
    
    # Загружаем тестовые данные
    if dataset_path:
        # Если передан путь к датасету, используем его
        if not Path(dataset_path).exists():
            print(f"❌ Файл {dataset_path} не найден")
            return
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Обрабатываем стандартный формат с metadata
        if isinstance(data, dict) and 'dialogs' in data:
            print("✅ Обнаружен стандартный формат. Загрузка...")
            
            source_dialogs = data.get('dialogs', [])
            test_data = []

            for dialog in source_dialogs:
                entities = []
                
                # Проверяем два возможных поля для сущностей: 'entities' и 'personal_data'
                entities_source = dialog.get('entities', dialog.get('personal_data', []))
                
                for entity_data in entities_source:
                    # Создаем стандартизированную сущность
                    entity = {
                        'start': entity_data.get('start'),
                        'end': entity_data.get('end'),
                        'type': entity_data.get('type'),
                    }
                    
                    # Проверяем разные поля для текста сущности
                    if 'text' in entity_data:
                        entity['text'] = entity_data['text']
                    elif 'value' in entity_data:
                        entity['text'] = entity_data['value']
                    else:
                        entity['text'] = ''
                    
                    entities.append(entity)

                test_data.append({
                    'id': dialog.get('id', len(test_data) + 1),
                    'text': dialog.get('text', ''),
                    'entities': entities,
                    'has_pd': bool(entities),
                    'description': dialog.get('description', f"Диалог ID: {dialog.get('id')}")
                })
        else:
            test_data = data
    else:
        test_data = load_test_dataset()
    
    if not test_data:
        print("❌ Нет тестовых данных")
        return
    
    print(f"📋 Загружено {len(test_data)} примеров")
    
    # Получаем название датасета
    dataset_name = get_dataset_name(dataset_path) if dataset_path else "test_dataset"
    print(f"📊 Название датасета: {dataset_name}")
    
    # Папка для сохранения отчетов
    output_dir = Path("data/reports")
    
    try:
        # Валидируем DeepPavlov
        overall_metrics, metrics_by_type = validate_deeppavlov(test_data, dataset_name)
        
        print(f"✅ DeepPavlov: F1={overall_metrics['f1']:.3f}, P={overall_metrics['precision']:.3f}, R={overall_metrics['recall']:.3f}")
        
        # Сохраняем индивидуальные метрики в отдельный CSV
        save_metrics_to_csv(metrics_by_type, "DeepPavlov", dataset_name, output_dir)
        
        files_created = 1
        
        # Если указан флаг update_pivot, пытаемся обновить сводную таблицу
        if update_pivot:
            print(f"\n🔄 Попытка обновления сводной таблицы...")
            if update_combined_metrics_csv(metrics_by_type, "DeepPavlov", dataset_name, output_dir):
                files_created = 2  # Обновили существующий файл
        
    except Exception as e:
        print(f"❌ Ошибка в DeepPavlov: {e}")
        return
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("📊 СТРОГИЕ МЕТРИКИ (EXACT MATCH)")
    print("=" * 60)
    
    print(f"Всего ожидаемых сущностей: {overall_metrics['tp'] + overall_metrics['fn']}")
    print(f"Всего найденных сущностей: {overall_metrics['tp'] + overall_metrics['fp']}")
    print(f"Правильно найденных: {overall_metrics['tp']}")
    print(f"")
    print(f"🎯 Precision: {overall_metrics['precision']:.3f} ({overall_metrics['precision']*100:.1f}%)")
    print(f"🎯 Recall: {overall_metrics['recall']:.3f} ({overall_metrics['recall']*100:.1f}%)")
    print(f"🎯 F1-Score: {overall_metrics['f1']:.3f} ({overall_metrics['f1']*100:.1f}%)")
    
    # Анализ производительности
    print(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ")
    print(f"Время инициализации: {overall_metrics['init_time']:.2f}с")
    print(f"Среднее время на пример: {overall_metrics['avg_time_per_example']*1000:.1f}мс")
    print(f"\n📁 Отчеты сохранены в папку: {output_dir}")
    print(f"📄 Всего файлов создано/обновлено: {files_created}")


def main():
    """Основная функция."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Валидация DeepPavlov анонимизатора",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python validate_deeppavlov.py
  python validate_deeppavlov.py --dataset test_dataset.json
  python validate_deeppavlov.py --dataset data/generated/deepseek_v3_dataset_manual_final.json
  python validate_deeppavlov.py -d /path/to/custom_dataset.json --update-pivot
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Путь к датасету для валидации. Если не указан, используется test_dataset.json'
    )
    
    parser.add_argument(
        '--update-pivot',
        action='store_true',
        help='Попытаться обновить существующую сводную таблицу, добавив данные DeepPavlov'
    )
    
    args = parser.parse_args()
    
    print("🚀 Улучшенная валидация DeepPavlov моделей")
    print("=" * 50)
    
    analyze_deeppavlov_improved(args.dataset, args.update_pivot)


if __name__ == "__main__":
    main() 