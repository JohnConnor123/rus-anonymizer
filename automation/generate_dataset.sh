#!/bin/bash

# Обертка для генерации датасета
# Использование: ./automation/generate_dataset.sh

set -e  # Остановка при ошибке

echo "🚀 Запуск генератора датасета"
echo "================================"

# Переходим в корневую папку проекта
cd "$(dirname "$0")/.."

# Проверяем виртуальное окружение
if [ ! -d "venv" ]; then
    echo "❌ Ошибка: виртуальное окружение venv не найдено"
    exit 1
fi

# Проверяем конфигурацию
if [ ! -f "config.env" ]; then
    echo "❌ Ошибка: файл config.env не найден"
    exit 1
fi

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source venv/bin/activate

# Проверяем зависимости
echo "📦 Проверка зависимостей..."
if ! python -c "import openai" 2>/dev/null; then
    echo "⚠️ Устанавливаю зависимости..."
    pip install -r requirements.txt
fi

# Запускаем генератор
echo "🎯 Запуск generate_dataset.py..."
python generate_dataset.py

echo "✅ Генерация завершена" 