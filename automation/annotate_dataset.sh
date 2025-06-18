#!/bin/bash

# Обертка для ручной аннотации датасета
# Использование: ./automation/annotate_dataset.sh <путь_к_датасету>

set -e

echo "✏️ Запуск аннотации датасета"
echo "============================="

# Переходим в корневую папку проекта
cd "$(dirname "$0")/.."

# Проверяем аргументы
if [ -z "$1" ]; then
    echo "❌ Ошибка: необходимо указать путь к датасету"
    echo "💡 Использование: ./automation/annotate_dataset.sh <путь_к_датасету>"
    exit 1
fi

DATASET_PATH="$1"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Ошибка: файл $DATASET_PATH не найден"
    exit 1
fi

# Проверяем виртуальное окружение
if [ ! -d "venv" ]; then
    echo "❌ Ошибка: виртуальное окружение venv не найдено"
    exit 1
fi

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source venv/bin/activate

# Создаем временный файл аннотации
ANNOTATION_SCRIPT="temp_annotate_$(date +%s).py"

echo "📝 Создание скрипта аннотации..."
cat > "$ANNOTATION_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Скрипт для ручной аннотации датасета
Автоматически сгенерирован automation/annotate_dataset.sh
"""

import json
import sys
from pathlib import Path

def find_entity_position(text, entity_text, start_hint=0):
    """Находит позицию сущности в тексте"""
    pos = text.find(entity_text, start_hint)
    if pos == -1:
        # Пробуем с нормализацией пробелов
        normalized_entity = ' '.join(entity_text.split())
        normalized_text = ' '.join(text.split())
        pos = normalized_text.find(normalized_entity, start_hint)
        if pos != -1:
            # Пересчитываем позицию в оригинальном тексте
            pos = text.find(entity_text.strip(), start_hint)
    return pos

def annotate_dataset(dataset_path):
    """Основная функция аннотации"""
    print(f"🔍 Загрузка датасета: {dataset_path}")
    
    # Здесь будет код аннотации
    # Этот скрипт будет дополнен при каждом запуске
    
    print("⚠️ Скрипт аннотации готов к использованию")
    print("💡 Добавьте сюда логику ручной разметки")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python script.py <путь_к_датасету>")
        sys.exit(1)
    
    annotate_dataset(sys.argv[1])
EOF

echo "🎯 Запуск аннотации для: $DATASET_PATH"
python "$ANNOTATION_SCRIPT" "$DATASET_PATH"

# Удаляем временный файл
rm -f "$ANNOTATION_SCRIPT"

echo "✅ Аннотация завершена" 