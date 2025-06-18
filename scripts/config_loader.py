#!/usr/bin/env python3
"""
Модуль для загрузки конфигурации из файла окружения
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Загрузчик конфигурации из файла окружения"""
    
    def __init__(self, config_file: str = "config.env"):
        """
        Инициализация загрузчика конфигурации
        
        Args:
            config_file: Путь к файлу конфигурации
        """
        self.config_file = config_file
        self.config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Загружает конфигурацию из файла"""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Файл конфигурации {self.config_file} не найден"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Пропускаем комментарии и пустые строки
                if not line or line.startswith('#'):
                    continue
                
                # Парсим переменные вида KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Удаляем inline комментарии после символа #, если они есть
                    if '#' in value:
                        value = value.split('#', 1)[0]
                    self.config[key.strip()] = value.strip()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение из конфигурации
        
        Args:
            key: Ключ конфигурации
            default: Значение по умолчанию
            
        Returns:
            Значение конфигурации или значение по умолчанию
        """
        return self.config.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """
        Получает целочисленное значение из конфигурации
        
        Args:
            key: Ключ конфигурации
            default: Значение по умолчанию
            
        Returns:
            Целочисленное значение
        """
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Получает булево значение из конфигурации
        
        Args:
            key: Ключ конфигурации
            default: Значение по умолчанию
            
        Returns:
            Булево значение
        """
        value = self.get(key, default)
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def get_str(self, key: str, default: str = "") -> str:
        """
        Получает строковое значение из конфигурации
        
        Args:
            key: Ключ конфигурации
            default: Значение по умолчанию
            
        Returns:
            Строковое значение
        """
        value = self.get(key, default)
        if value == "" or value is None:
            return default
        return str(value)
    
    def print_config(self) -> None:
        """Выводит текущую конфигурацию"""
        print("🔧 Текущая конфигурация:")
        for key, value in self.config.items():
            # Скрываем API ключи
            if 'key' in key.lower() or 'token' in key.lower():
                display_value = "***скрыто***" if value else "не установлено"
            else:
                display_value = value
            print(f"   {key}: {display_value}")


def load_dataset_config() -> Dict[str, Any]:
    """Загружает конфигурацию для генерации датасета из файла config.env"""
    
    loader = ConfigLoader()
    
    config = {
        # Основные параметры генерации
        'total_dialogs': loader.get_int('TOTAL_DIALOGS', 100),
        'max_batch_size': loader.get_int('MAX_BATCH_SIZE', 20),
        'output_file': loader.get_str('OUTPUT_FILE', None),
        
        # OpenAI API параметры
        'api_key': loader.get_str('OPENAI_API_KEY'),
        'model': loader.get_str('MODEL', 'meta-llama/llama-4-maverick'),
        'base_url': loader.get_str('OPENAI_BASE_URL', None),
        
        # Режимы работы
        'preview_config': loader.get_bool('PREVIEW_CONFIG', False),
        'dry_run': loader.get_bool('DRY_RUN', False),
        'async_generation': loader.get_bool('ASYNC_GENERATION', True),
        'max_concurrency': loader.get_int('MAX_CONCURRENCY', 3)
    }
    
    return config 