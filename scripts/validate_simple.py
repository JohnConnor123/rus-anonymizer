#!/usr/bin/env python3
"""
Простая валидация анонимизаторов без сохранения отчетов.
Полезно для быстрых проверок и будущих прогонов.
"""

import json
import time
import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import os

# Импорт прогрессбара
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("📦 tqdm не установлен. Установите: pip install tqdm")
    TQDM_AVAILABLE = False

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импорты анонимизаторов
from anonymizers.natasha_enhanced.enhanced_natasha import EnhancedNatashaAnonymizer
from anonymizers.spacy_extended.spacy_anonymizer import SpacyAnonymizer
from anonymizers.transformers_bert.bert_anonymizer import BertAnonymizer
from anonymizers.hybrid_advanced.hybrid_anonymizer import HybridAdvancedAnonymizer
from anonymizers.regexp_baseline.regexp_anonymizer import RegExpBaselineAnonymizer

# Импорт обновлённого модуля метрик
from utils.metrics import calculate_metrics, calculate_metrics_by_type


def load_test_dataset(dataset_path: str = None) -> List[Dict]:
    """
    Загружает тестовый датасет из JSON файла.
    Поддерживает стандартный формат с metadata и dialogs.
    """
    test_file_path = dataset_path
    
    print(f"📂 Загрузка датасета: {test_file_path}")
    
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка чтения файла {test_file_path}: {e}")
        return []

    # Обрабатываем стандартный формат с metadata
    if isinstance(data, dict) and 'dialogs' in data:
        print("✅ Обнаружен стандартный формат. Загрузка...")
        
        source_dialogs = data.get('dialogs', [])
        converted_dialogs = []

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

            converted_dialogs.append({
                'id': dialog.get('id', len(converted_dialogs) + 1),
                'text': dialog.get('text', ''),
                'entities': entities,
                'has_pd': bool(entities),
                'description': dialog.get('description', f"Диалог ID: {dialog.get('id')}")
            })
        
        print(f"👍 Загрузка завершена. {len(converted_dialogs)} диалогов готово к валидации.")
        return converted_dialogs

    print(f"❌ Неизвестный формат датасета в файле {test_file_path}")
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


def improve_extracted_entities(extracted: List[Tuple[int, int, str, str]], text: str) -> List[Tuple[int, int, str, str]]:
    """Улучшает извлеченные сущности."""
    # Объединяем соседние PERSON сущности
    improved = merge_person_entities(extracted)
    
    # Улучшаем AGE сущности - убираем лишние слова
    final_improved = []
    for start, end, entity_type, value in improved:
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
                    final_improved.append((age_start, age_end, entity_type, age_text))
                    continue
        
        final_improved.append((start, end, entity_type, value))
    
    return final_improved


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


def save_combined_metrics_to_csv(all_results: Dict[str, Dict[str, Dict[str, float]]], 
                                dataset_name: str, output_dir: Path):
    """Сохраняет объединенные метрики всех анонимизаторов в один CSV файл."""
    filename = f"{dataset_name}_combined_metrics.csv"
    filepath = output_dir / filename
    
    # Создаем директорию если не существует
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['method', 'entity_type', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'total_true', 'total_pred', 'class_proportion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for method_name, metrics_by_type in all_results.items():
            for entity_type, metrics in metrics_by_type.items():
                row = {
                    'method': method_name,
                    'entity_type': entity_type
                }
                row.update(metrics)
                writer.writerow(row)
    
    print(f"💾 Сводные метрики сохранены в {filepath}")


def validate_anonymizer(anonymizer, name: str, test_data: List[Dict], verbose: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Валидирует один анонимизатор и возвращает общие метрики и метрики по типам."""
    if verbose:
        print(f"\n🔍 Валидация {name}...")
    else:
        print(f"Валидация {name}...")
    
    start_time = time.time()
    
    all_true_entities = []
    all_predicted_entities = []
    processing_times = []
    
    # Настройка прогрессбара
    if TQDM_AVAILABLE:
        iterator = tqdm(enumerate(test_data), total=len(test_data), desc=f"{name}", leave=False)
    else:
        iterator = enumerate(test_data)
    
    for i, example in iterator:
        text = example['text']
        true_entities = example['entities']
        
        # Измеряем время обработки
        process_start = time.time()
        
        try:
            # Извлекаем сущности
            extracted = anonymizer.extract_entities(text)
            
            # Улучшаем извлеченные сущности
            improved_extracted = improve_extracted_entities(extracted, text)
            
            # Преобразуем в формат для сравнения
            predicted_entities = []
            for start, end, entity_type, value in improved_extracted:
                predicted_entities.append({
                    'start': start,
                    'end': end,
                    'type': entity_type,
                    'value': value
                })
            
            all_true_entities.extend(true_entities)
            all_predicted_entities.extend(predicted_entities)
            
        except Exception as e:
            error_emoji = "❌ " if verbose else ""
            print(f"{error_emoji}Ошибка в примере {i+1}: {e}")
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
        'avg_time_per_example': sum(processing_times) / len(processing_times) if processing_times else 0,
        'examples_processed': len(processing_times)
    })
    
    return overall_metrics, metrics_by_type


def print_results(results: Dict[str, Dict[str, Any]], all_detailed_results: Dict[str, Dict[str, Dict[str, float]]], verbose: bool = False):
    """Выводит результаты в консоль."""
    print("\n" + "="*80)
    if verbose:
        print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (СТРОГИЕ МЕТРИКИ)")
    else:
        print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (СТРОГИЕ МЕТРИКИ)")
    print("="*80)
    
    # Заголовок таблицы
    print(f"{'Анонимизатор':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Время (с)':<12}")
    print("-" * 80)
    
    # Сортируем по F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1']:<10.3f} {metrics['total_time']:<12.2f}")
    
    print("-" * 80)
    
    # Детальная информация о лучшем результате
    best_name, best_metrics = sorted_results[0]
    best_emoji = "🏆 " if verbose else ""
    print(f"\n{best_emoji}Лучший результат: {best_name}")
    print(f"   F1-Score: {best_metrics['f1']:.3f}")
    print(f"   Precision: {best_metrics['precision']:.3f}")
    print(f"   Recall: {best_metrics['recall']:.3f}")
    print(f"   TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, FN: {best_metrics['fn']}")
    print(f"   Среднее время на пример: {best_metrics['avg_time_per_example']*1000:.1f}мс")
    
    # Добавляем macro метрики для каждого метода
    print("\n" + "="*80)
    if verbose:
        print("📊 MACRO МЕТРИКИ ДЛЯ КАЖДОГО МЕТОДА")
    else:
        print("MACRO МЕТРИКИ ДЛЯ КАЖДОГО МЕТОДА")
    print("="*80)
    
    print(f"{'Анонимизатор':<25} {'Macro P':<10} {'Macro R':<10} {'Macro F1':<10}")
    print("-" * 80)
    
    for name, metrics in sorted_results:
        if name in all_detailed_results and 'ALL_MACRO' in all_detailed_results[name]:
            macro_metrics = all_detailed_results[name]['ALL_MACRO']
            print(f"{name:<25} {macro_metrics['precision']:<10.3f} {macro_metrics['recall']:<10.3f} "
                  f"{macro_metrics['f1']:<10.3f}")
        else:
            print(f"{name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print("-" * 80)


def main():
    """Основная функция валидации."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Простая валидация анонимизаторов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python validate_simple.py --dataset test_dataset.json
  python validate_simple.py --dataset data/generated/deepseek_v3_dataset_manual_final.json --verbose
  python validate_simple.py -d /path/to/custom_dataset.json -v
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Путь к датасету для валидации.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Включить подробный вывод с эмодзи и деталями (по умолчанию отключен)'
    )
    
    args = parser.parse_args()
    
    startup_emoji = "🚀 " if args.verbose else ""
    print(f"{startup_emoji}Запуск простой валидации анонимизаторов")
    
    folder_emoji = "📁 " if args.verbose else ""
    print(f"{folder_emoji}Рабочая папка: {Path.cwd()}")
    
    # Загружаем тестовые данные
    test_data = load_test_dataset(args.dataset)
    if not test_data:
        error_emoji = "❌ " if args.verbose else ""
        print(f"{error_emoji}Нет тестовых данных для валидации")
        return
    
    list_emoji = "📋 " if args.verbose else ""
    print(f"{list_emoji}Загружено {len(test_data)} примеров для тестирования")
    
    # Получаем название датасета
    dataset_name = get_dataset_name(args.dataset)
    chart_emoji = "📊 " if args.verbose else ""
    print(f"{chart_emoji}Название датасета: {dataset_name}")
    
    # Папка для сохранения отчетов
    output_dir = Path("data/reports")
    
    # Список анонимизаторов для тестирования
    anonymizers = {
        "RegExp Baseline": RegExpBaselineAnonymizer(),
        "Natasha Enhanced": EnhancedNatashaAnonymizer(),
        "SpaCy Extended": SpacyAnonymizer(),
        "Transformers BERT": BertAnonymizer(),
        "Hybrid Advanced": HybridAdvancedAnonymizer(verbose=args.verbose),
    }
    
    target_emoji = "🎯 " if args.verbose else ""
    print(f"{target_emoji}Загружено {len(anonymizers)} анонимизаторов")
    
    # Валидируем каждый анонимизатор
    results = {}
    all_detailed_results = {}
    
    for name, anonymizer in anonymizers.items():
        try:
            overall_metrics, metrics_by_type = validate_anonymizer(anonymizer, name, test_data, args.verbose)
            results[name] = overall_metrics
            all_detailed_results[name] = metrics_by_type
            
            check_emoji = "✅ " if args.verbose else ""
            print(f"{check_emoji}{name}: F1={overall_metrics['f1']:.3f}, P={overall_metrics['precision']:.3f}, R={overall_metrics['recall']:.3f}")
            
            # Сохраняем метрики анонимизатора в отдельный CSV
            save_metrics_to_csv(metrics_by_type, name, dataset_name, output_dir)
            
        except Exception as e:
            error_emoji = "❌ " if args.verbose else ""
            print(f"{error_emoji}Ошибка в {name}: {e}")
            continue
    
    # Сохраняем объединенные метрики
    if all_detailed_results:
        save_combined_metrics_to_csv(all_detailed_results, dataset_name, output_dir)
    
    # Выводим итоговые результаты
    if results:
        print_results(results, all_detailed_results, args.verbose)
        folder_emoji = "📁 " if args.verbose else ""
        print(f"\n{folder_emoji}Отчеты сохранены в папку: {output_dir}")
        file_emoji = "📄 " if args.verbose else ""
        print(f"{file_emoji}Всего файлов создано: {len(anonymizers) + 1}")
    else:
        error_emoji = "❌ " if args.verbose else ""
        print(f"{error_emoji}Не удалось получить результаты валидации")


if __name__ == "__main__":
    main() 