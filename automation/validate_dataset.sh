#!/bin/bash

# Обертка для валидации датасета
# Использование: ./automation/validate_dataset.sh [путь_к_датасету]

set -e

echo "🔍 Запуск валидации анонимизаторов"
echo "=================================="

# Переходим в корневую папку проекта
cd "$(dirname "$0")/.."

# Проверяем виртуальное окружение
if [ ! -d "venv" ]; then
    echo "❌ Ошибка: виртуальное окружение venv не найдено"
    exit 1
fi

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source venv/bin/activate

# Проверяем зависимости
echo "📦 Проверка зависимостей..."
if ! python -c "import spacy" 2>/dev/null; then
    echo "⚠️ Устанавливаю зависимости..."
    pip install -r requirements.txt
fi

# Запускаем валидацию
echo "🎯 Запуск валидации..."
if [ -n "$1" ]; then
    echo "📁 Используем датасет: $1"
    python tests/validate_simple.py --dataset "$1"
else
    echo "📁 Используем стандартный датасет"
    python tests/validate_simple.py
fi

echo "✅ Валидация завершена" 