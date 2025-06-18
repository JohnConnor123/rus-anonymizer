#!/usr/bin/env python3
"""
Автоматическая разметка датасета llama-4-maverick-2.json
Находит пропущенные персональные данные и добавляет их в entities
"""

import json
import re
from typing import List, Dict, Tuple, Any
from pathlib import Path

# Регулярные выражения для поиска ПД
PD_PATTERNS = {
    'PHONE': [
        re.compile(r"(?:\+7|8)?\s?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}"),
        re.compile(r"8\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}"),
        re.compile(r"\+7\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}")
    ],
    'EMAIL': [
        re.compile(r"[a-zA-Z0-9._%+-]+@[\w.-]+\.[a-zA-Z]{2,}")
    ],
    'PERSON': [
        re.compile(r"(?:меня зовут|зовут меня|моё имя|мое имя)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)", re.IGNORECASE),
        re.compile(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s+(?:звонит|звоню|обращается)", re.IGNORECASE)
    ],
    'PASSPORT': [
        re.compile(r"\b\d{4}\s?\d{6}\b"),
        re.compile(r"паспорт\s*:?\s*(\d{4}\s?\d{6})", re.IGNORECASE)
    ],
    'SNILS': [
        re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}\b"),
        re.compile(r"СНИЛС\s*:?\s*(\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2})", re.IGNORECASE)
    ],
    'INN': [
        re.compile(r"\b\d{10,12}\b"),
        re.compile(r"ИНН\s*:?\s*(\d{10,12})", re.IGNORECASE)
    ],
    'CARD_NUMBER': [
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b"),
        re.compile(r"карт[аы]\s*:?\s*(\d{4}\s\d{4}\s\d{4}\s\d{4})", re.IGNORECASE)
    ],
    'VIN_NUMBER': [
        re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
        re.compile(r"VIN\s*:?\s*([A-HJ-NPR-Z0-9]{17})", re.IGNORECASE)
    ],
    'BANK_ACCOUNT': [
        re.compile(r"\b\d{20}\b"),
        re.compile(r"расчетный счет\s*:?\s*(\d{20})", re.IGNORECASE)
    ]
}

def find_entities_in_text(text: str) -> List[Dict[str, Any]]:
    """
    Находит все сущности в тексте с помощью регулярных выражений
    
    Args:
        text: Текст для анализа
        
    Returns:
        Список найденных сущностей
    """
    entities = []
    
    for entity_type, patterns in PD_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                # Для именованных групп берем группу 1, иначе всё совпадение
                if match.groups():
                    entity_text = match.group(1)
                    start = match.start(1)
                    end = match.end(1)
                else:
                    entity_text = match.group(0)
                    start = match.start()
                    end = match.end()
                
                # Дополнительная валидация
                if validate_entity(entity_type, entity_text):
                    entities.append({
                        'start': start,
                        'end': end,
                        'type': entity_type,
                        'text': entity_text
                    })
    
    # Удаляем дубликаты (одинаковые позиции)
    unique_entities = []
    seen_positions = set()
    
    for entity in entities:
        pos_key = (entity['start'], entity['end'])
        if pos_key not in seen_positions:
            unique_entities.append(entity)
            seen_positions.add(pos_key)
    
    return unique_entities

def validate_entity(entity_type: str, text: str) -> bool:
    """
    Дополнительная валидация найденных сущностей
    
    Args:
        entity_type: Тип сущности
        text: Текст сущности
        
    Returns:
        True если сущность валидна
    """
    text = text.strip()
    
    if entity_type == 'PHONE':
        # Убираем все нецифровые символы и проверяем длину
        digits = re.sub(r'\D', '', text)
        return len(digits) in [10, 11] and digits.isdigit()
    
    elif entity_type == 'EMAIL':
        # Базовая проверка email
        return '@' in text and '.' in text.split('@')[1]
    
    elif entity_type == 'PERSON':
        # Имя должно содержать только буквы и пробелы
        return re.match(r'^[А-ЯЁа-яё\s]+$', text) and len(text.split()) >= 1
    
    elif entity_type == 'PASSPORT':
        # Паспорт: 4 цифры + 6 цифр
        digits = re.sub(r'\D', '', text)
        return len(digits) == 10 and digits.isdigit()
    
    elif entity_type == 'SNILS':
        # СНИЛС: 11 цифр
        digits = re.sub(r'\D', '', text)
        return len(digits) == 11 and digits.isdigit()
    
    elif entity_type == 'INN':
        # ИНН: 10 или 12 цифр
        return len(text) in [10, 12] and text.isdigit()
    
    elif entity_type == 'CARD_NUMBER':
        # Номер карты: 16 цифр
        digits = re.sub(r'\D', '', text)
        return len(digits) == 16 and digits.isdigit()
    
    elif entity_type == 'VIN_NUMBER':
        # VIN: 17 символов, исключая I, O, Q
        return len(text) == 17 and not any(c in text for c in 'IOQ')
    
    elif entity_type == 'BANK_ACCOUNT':
        # Расчетный счет: 20 цифр
        return len(text) == 20 and text.isdigit()
    
    return True

def annotate_dataset(input_file: str, output_file: str) -> Dict[str, Any]:
    """
    Аннотирует датасет, добавляя пропущенные сущности
    
    Args:
        input_file: Путь к исходному датасету
        output_file: Путь к аннотированному датасету
        
    Returns:
        Статистика аннотации
    """
    print(f"📂 Загрузка датасета: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dialogs = data.get('dialogs', [])
    print(f"📋 Загружено {len(dialogs)} диалогов")
    
    # Статистика
    stats = {
        'total_dialogs': len(dialogs),
        'originally_with_pd': sum(1 for d in dialogs if d['has_pd']),
        'originally_without_pd': sum(1 for d in dialogs if not d['has_pd']),
        'dialogs_updated': 0,
        'entities_added': 0,
        'entity_types_found': {}
    }
    
    # Аннотируем каждый диалог
    for dialog in dialogs:
        text = dialog['text']
        existing_entities = dialog.get('entities', [])
        
        # Находим новые сущности
        found_entities = find_entities_in_text(text)
        
        # Проверяем какие сущности уже есть
        existing_positions = {(e['start'], e['end']) for e in existing_entities}
        new_entities = [e for e in found_entities 
                       if (e['start'], e['end']) not in existing_positions]
        
        if new_entities:
            # Добавляем новые сущности
            dialog['entities'].extend(new_entities)
            dialog['has_pd'] = True
            
            stats['dialogs_updated'] += 1
            stats['entities_added'] += len(new_entities)
            
            # Подсчитываем типы сущностей
            for entity in new_entities:
                entity_type = entity['type']
                stats['entity_types_found'][entity_type] = (
                    stats['entity_types_found'].get(entity_type, 0) + 1
                )
            
            print(f"✅ Диалог {dialog['id']}: добавлено {len(new_entities)} сущностей")
            for entity in new_entities:
                print(f"   - {entity['type']}: '{entity['text']}'")
    
    # Обновляем статистику датасета
    stats['finally_with_pd'] = sum(1 for d in dialogs if d['has_pd'])
    stats['finally_without_pd'] = len(dialogs) - stats['finally_with_pd']
    
    # Сохраняем аннотированный датасет
    data['metadata']['annotation_stats'] = stats
    data['metadata']['annotation_timestamp'] = __import__('time').time()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Аннотированный датасет сохранен: {output_file}")
    
    return stats

def print_annotation_stats(stats: Dict[str, Any]):
    """Выводит статистику аннотации"""
    print("\n" + "="*60)
    print("📊 СТАТИСТИКА АННОТАЦИИ")
    print("="*60)
    
    print(f"📋 Всего диалогов: {stats['total_dialogs']}")
    print(f"📝 Обновлено диалогов: {stats['dialogs_updated']}")
    print(f"🆕 Добавлено сущностей: {stats['entities_added']}")
    
    print(f"\n📈 Изменение распределения ПД:")
    print(f"   Было с ПД: {stats['originally_with_pd']} ({stats['originally_with_pd']/stats['total_dialogs']*100:.1f}%)")
    print(f"   Стало с ПД: {stats['finally_with_pd']} ({stats['finally_with_pd']/stats['total_dialogs']*100:.1f}%)")
    print(f"   Прирост: +{stats['finally_with_pd'] - stats['originally_with_pd']} диалогов")
    
    if stats['entity_types_found']:
        print(f"\n🔍 Найденные типы сущностей:")
        for entity_type, count in sorted(stats['entity_types_found'].items()):
            print(f"   {entity_type}: {count}")

def main():
    """Основная функция"""
    input_file = "data/generated/llama-4-maverick-2.json"
    output_file = "data/generated/llama-4-maverick-2_annotated.json"
    
    if not Path(input_file).exists():
        print(f"❌ Файл {input_file} не найден")
        return 1
    
    print("🚀 Автоматическая аннотация датасета")
    print("="*50)
    
    try:
        stats = annotate_dataset(input_file, output_file)
        print_annotation_stats(stats)
        
        print(f"\n🎉 Аннотация завершена успешно!")
        print(f"📄 Исходный файл: {input_file}")
        print(f"📄 Аннотированный файл: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Ошибка аннотации: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 