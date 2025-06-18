#!/bin/bash

# Обертка для анализа качества аннотации
# Использование: ./automation/analyze_quality.sh <путь_к_датасету>

set -e

echo "📊 Запуск анализа качества аннотации"
echo "===================================="

# Переходим в корневую папку проекта
cd "$(dirname "$0")/.."

# Проверяем аргументы
if [ -z "$1" ]; then
    echo "❌ Ошибка: необходимо указать путь к датасету"
    echo "💡 Использование: ./automation/analyze_quality.sh <путь_к_датасету>"
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

# Проверяем наличие скрипта анализа
if [ ! -f "scripts/analyze_annotation_quality.py" ]; then
    echo "❌ Ошибка: скрипт анализа не найден"
    exit 1
fi

# Запускаем анализ
echo "🎯 Запуск анализа качества для: $DATASET_PATH"
python scripts/analyze_annotation_quality.py "$DATASET_PATH"

echo "✅ Анализ завершен" 