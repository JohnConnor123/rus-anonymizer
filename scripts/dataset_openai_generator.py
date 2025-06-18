#!/usr/bin/env python3
"""
Генератор датасета для анонимизации ПД через OpenAI API
"""

import json
import time
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import openai
from openai import OpenAI
from openai import AsyncOpenAI
import logging

from .dataset_config import (
    sample_random_config, 
    generate_batch_configs, 
    OPENAI_CONFIG, 
    DEFAULT_GENERATION_PARAMS
)
from .dataset_prompts import format_prompt

import asyncio

# Компиляция базовых regex для быстрой проверки наличия ПД
PD_REGEXPS = [
    re.compile(r"(?:\+7|8)?\s?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}"),  # PHONE
    re.compile(r"[a-zA-Z0-9._%+-]+@[\w.-]+\.[a-zA-Z]{2,}"),  # EMAIL
    re.compile(r"\b\d{4}\s?\d{6}\b"),  # PASSPORT simple
    re.compile(r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b")  # CARD number
]

# Настроим простой логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenAIDatasetGenerator:
    """Генератор датасета через OpenAI API с случайным семплированием параметров"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None, base_url: Optional[str] = None):
        """
        Инициализация генератора
        
        Args:
            api_key: OpenAI API ключ (если None, то берется из переменных окружения)
            model: Модель для генерации (по умолчанию из конфига)
            base_url: Адрес API эндпоинта
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model or OPENAI_CONFIG["model"]
        self.max_tokens = OPENAI_CONFIG["max_tokens"]
        self.temperature = OPENAI_CONFIG["temperature"]
        self.timeout = OPENAI_CONFIG["timeout"]
        
        # Статистика генерации
        self.stats = {
            "total_generated": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_cost_tokens": 0,
            "generation_configs": []
        }
        
    def generate_single_batch(self, config: Dict[str, Any]) -> List[Dict]:
        """
        Генерирует одну пачку диалогов с заданной конфигурацией
        
        Args:
            config: Конфигурация генерации (из sample_random_config)
            
        Returns:
            Список диалогов в формате test_dataset.json
        """
        print(f"🚀 Генерация пачки: {config['batch_size']} диалогов")
        print(f"📍 Сфера: {config['business_sphere']}")
        print(f"🌍 Регион: {config['regional_focus']}")
        print(f"📞 Стиль: {config['communication_style']} | Качество: {config['transcription_quality']}")
        
        # Формируем промпт
        prompt_config = config.copy()
        prompt_config["total_dialogs"] = config["batch_size"]
        prompt = format_prompt(prompt_config)
        
        try:
            # Запрос к OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "Ты эксперт по генерации данных для тестирования систем анонимизации персональных данных. Если пропущена хотя бы одна сущность — ответ аннулируется и ты не получаешь оплату. Отвечай ТОЛЬКО валидным JSON массивом без markdown."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                response_format={"type": "json_array"}
            )
            
            # Парсим ответ
            content = response.choices[0].message.content.strip()
            dialogs = self._parse_response(content)
            
            # Обновляем статистику
            self.stats["total_generated"] += len(dialogs)
            self.stats["successful_batches"] += 1
            self.stats["total_cost_tokens"] += response.usage.total_tokens
            self.stats["generation_configs"].append(config)
            
            print(f"✅ Успешно сгенерировано {len(dialogs)} диалогов")
            print(f"💰 Потрачено токенов: {response.usage.total_tokens}")
            
            return dialogs
            
        except Exception as e:
            print(f"❌ Ошибка генерации пачки: {e}")
            self.stats["failed_batches"] += 1
            return []
    
    def _parse_response(self, content: str, allow_repair: bool = True) -> List[Dict]:
        """
        Парсит ответ от OpenAI в формат test_dataset.json
        
        Args:
            content: Текст ответа от модели
            allow_repair: Разрешить исправление ответа
            
        Returns:
            Список диалогов
        """
        try:
            # Убираем markdown форматирование если есть
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*$', '', content)
            content = content.strip()
            
            # Парсим JSON
            dialogs = json.loads(content)
            
            # Проверяем формат
            if not isinstance(dialogs, list):
                raise ValueError("Ответ должен быть JSON массивом")
                
            # Валидируем и нормализуем каждый диалог
            normalized_dialogs = []
            for i, dialog in enumerate(dialogs):
                normalized_dialog = self._normalize_dialog(dialog, i + 1)
                if normalized_dialog:
                    normalized_dialogs.append(normalized_dialog)
                    
            missing = [d for d in normalized_dialogs if self._has_potential_pd(d["text"]) and len(d["entities"]) == 0]
            if missing and allow_repair:
                logger.info("Обнаружено %d диалогов без разметки — пытаюсь исправить через follow-up", len(missing))
                repaired = self._repair_missing_entities(content)
                if repaired:
                    return repaired
                    
            return normalized_dialogs
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"Первые 500 символов ответа: {content[:500]}")
            return []
        except Exception as e:
            print(f"❌ Ошибка обработки ответа: {e}")
            return []
    
    def _normalize_dialog(self, dialog: Dict, default_id: int) -> Optional[Dict]:
        """
        Нормализует диалог в формат test_dataset.json
        
        Args:
            dialog: Исходный диалог от модели
            default_id: ID по умолчанию если отсутствует
            
        Returns:
            Нормализованный диалог или None если невалидный
        """
        try:
            # Обязательные поля
            if "text" not in dialog:
                print(f"⚠️ Пропущен диалог без поля 'text'")
                return None
                
            normalized = {
                "text": dialog["text"],
                "entities": dialog.get("entities", []),
                "has_pd": bool(dialog.get("has_pd", len(dialog.get("entities", [])) > 0)),
                "description": dialog.get("description", "Сгенерированный диалог"),
                "id": dialog.get("id", default_id)
            }
            
            # Валидируем entities
            validated_entities = []
            for entity in normalized["entities"]:
                if self._validate_entity(entity, normalized["text"]):
                    validated_entities.append(entity)
                    
            normalized["entities"] = validated_entities
            
            # Обновляем has_pd на основе наличия валидных entities
            normalized["has_pd"] = len(validated_entities) > 0
            
            return normalized
            
        except Exception as e:
            print(f"⚠️ Ошибка нормализации диалога: {e}")
            return None
    
    def _validate_entity(self, entity: Dict, text: str) -> bool:
        """
        Валидирует сущность
        
        Args:
            entity: Сущность для валидации
            text: Текст диалога
            
        Returns:
            True если сущность валидна
        """
        required_fields = ["start", "end", "type", "text"]
        
        # Проверяем наличие обязательных полей
        for field in required_fields:
            if field not in entity:
                return False
                
        # Проверяем корректность индексов
        start, end = entity["start"], entity["end"]
        if not (0 <= start < end <= len(text)):
            return False
            
        # Проверяем соответствие текста
        extracted_text = text[start:end]
        if extracted_text != entity["text"]:
            # Попробуем исправить незначительные расхождения
            if extracted_text.strip() == entity["text"].strip():
                entity["text"] = extracted_text  # Исправляем
                return True
            return False
            
        return True
    
    def _has_potential_pd(self, text: str) -> bool:
        """Быстрая эвристическая проверка наличия ПД по regex."""
        return any(r.search(text) for r in PD_REGEXPS)

    def _repair_missing_entities(self, original_json: str) -> List[Dict]:
        """Запрашивает у LLM исправление пустых entities."""
        try:
            repair_prompt = (
                "В некоторых диалогах не размечены сущности, хотя персональные данные присутствуют. "
                "Исправь ТОЛЬКО поля \"entities\" там, где они пусты. "
                "Верни полный JSON массив диалогов без пояснений и markdown."
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Тебе нужно исправить пропущенные сущности. Отвечай только JSON массивом."},
                    {"role": "user", "content": repair_prompt},
                    {"role": "user", "content": original_json}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                response_format={"type": "json_array"}
            )
            new_content = response.choices[0].message.content.strip()
            return self._parse_response(new_content, allow_repair=False)
        except Exception as exc:
            logger.error("Не удалось выполнить автодовалидацию: %s", exc)
            return []
    
    def generate_dataset(
        self,
        total_dialogs: int = None,
        max_batch_size: int = None,
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        Генерирует полный датасет с заданными параметрами
        
        Args:
            total_dialogs: Общее количество диалогов для генерации
            max_batch_size: Максимальный размер пачки
            save_path: Путь для сохранения результата
            
        Returns:
            Статистика генерации
        """
        # Используем настройки по умолчанию
        total_dialogs = total_dialogs or DEFAULT_GENERATION_PARAMS["total_dialogs"]
        max_batch_size = max_batch_size or DEFAULT_GENERATION_PARAMS["max_batch_size"]
        save_path = save_path or f"generated_dataset_{int(time.time())}.json"
        
        print(f"🎯 Генерация датасета: {total_dialogs} диалогов, размер пачки {max_batch_size}")
        
        # Генерируем конфигурации пачек
        batch_configs = generate_batch_configs(total_dialogs, max_batch_size)
        print(f"📦 Будет сгенерировано {len(batch_configs)} пачек")
        
        all_dialogs = []
        
        # Генерируем пачки последовательно
        for i, config in enumerate(batch_configs):
            print(f"\n--- Пачка {i + 1}/{len(batch_configs)} ---")
            
            batch_dialogs = self.generate_single_batch(config)
            all_dialogs.extend(batch_dialogs)
            
            # Небольшая пауза между запросами
            if i < len(batch_configs) - 1:
                time.sleep(1)
        
        # Присваиваем ID диалогам
        for i, dialog in enumerate(all_dialogs):
            dialog["id"] = i + 1
            
        print(f"\n📊 Общая статистика:")
        print(f"   Всего диалогов: {len(all_dialogs)}")
        
        with_pd = sum(1 for d in all_dialogs if d.get("has_pd"))
        without_pd = len(all_dialogs) - with_pd
        
        print(f"   С ПД: {with_pd} ({with_pd/len(all_dialogs)*100:.1f}%)")
        print(f"   Без ПД: {without_pd} ({without_pd/len(all_dialogs)*100:.1f}%)")
        
        # Анализ по сферам
        sphere_stats = {}
        for config in self.stats["generation_configs"]:
            sphere = config["business_sphere"]
            sphere_stats[sphere] = sphere_stats.get(sphere, 0) + config["batch_size"]
        
        print(f"\n🏢 Распределение по сферам:")
        for sphere, count in sphere_stats.items():
            print(f"   {sphere}: {count}")
        
        # Сохраняем датасет в стандартном формате с метаданными
        dataset = {
            "metadata": {
                "total_dialogs": len(all_dialogs),
                "generation_timestamp": time.time(),
                "model_used": self.model,
                "generation_stats": self.stats
            },
            "dialogs": all_dialogs
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
                
        print(f"\n✅ Датасет сохранен: {save_path}")
        
        # Возвращаем статистику
        return {
            "total_dialogs": len(all_dialogs),
            "with_pd": with_pd,
            "without_pd": without_pd,
            "successful_batches": self.stats["successful_batches"],
            "failed_batches": self.stats["failed_batches"],
            "total_cost_tokens": self.stats["total_cost_tokens"],
            "sphere_distribution": sphere_stats,
            "save_path": save_path
        }

    async def generate_single_batch_async(self, config: Dict[str, Any]) -> List[Dict]:
        """Асинхронная версия generate_single_batch"""
        print(f"🚀[async] Генерация пачки: {config['batch_size']} диалогов")
        prompt_config = config.copy()
        prompt_config["total_dialogs"] = config["batch_size"]
        prompt = format_prompt(prompt_config)

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по генерации данных для тестирования систем анонимизации персональных данных. Если пропущена хотя бы одна сущность — ответ аннулируется и ты не получаешь оплату. Отвечай ТОЛЬКО валидным JSON массивом, никакого дополнительного текста!"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                response_format={"type": "json_array"}
            )
            content = response.choices[0].message.content.strip()
            dialogs = self._parse_response(content)
            # Обновляем статистику (потом суммируем)
            batch_stats = {
                "dialogs": dialogs,
                "tokens": response.usage.total_tokens,
                "config": config
            }
            return batch_stats
        except Exception as e:
            print(f"❌ Ошибка async генерации пачки: {e}")
            return {"dialogs": [], "tokens": 0, "config": config, "failed": True}

    async def generate_dataset_async(self,
                                   total_dialogs: int = None,
                                   max_batch_size: int = None,
                                   save_path: str = None,
                                   max_concurrency: int = 3) -> Dict[str, Any]:
        """Асинхронная версия генерации датасета с параллельными запросами."""
        # Используем настройки по умолчанию
        total_dialogs = total_dialogs or DEFAULT_GENERATION_PARAMS["total_dialogs"]
        max_batch_size = max_batch_size or DEFAULT_GENERATION_PARAMS["max_batch_size"]
        save_path = save_path or f"generated_dataset_{int(time.time())}.json"

        print(f"🎯[async] Генерация датасета: {total_dialogs} диалогов, пачка {max_batch_size}, concurrency={max_concurrency}")

        batch_configs = generate_batch_configs(total_dialogs, max_batch_size)
        print(f"📦 Будет запущено {len(batch_configs)} параллельных задач")

        sem = asyncio.Semaphore(max_concurrency)

        async def sem_task(cfg):
            async with sem:
                return await self.generate_single_batch_async(cfg)

        tasks = [asyncio.create_task(sem_task(cfg)) for cfg in batch_configs]
        results = await asyncio.gather(*tasks)

        all_dialogs: List[Dict] = []
        sphere_stats: Dict[str, int] = {}
        for res in results:
            dialogs = res["dialogs"] if isinstance(res, dict) else []
            all_dialogs.extend(dialogs)
            if not res.get("failed"):
                self.stats["successful_batches"] += 1
            else:
                self.stats["failed_batches"] += 1
            self.stats["total_cost_tokens"] += res.get("tokens", 0)
            self.stats["generation_configs"].append(res["config"])
            sphere = res["config"]["business_sphere"]
            sphere_stats[sphere] = sphere_stats.get(sphere, 0) + res["config"]["batch_size"]

        # Присвоим ID
        for i, dialog in enumerate(all_dialogs):
            dialog["id"] = i + 1

        print(f"📊[async] Сгенерировано {len(all_dialogs)} диалогов")

        # Сохранение в стандартном формате с метаданными
        with_pd = sum(1 for d in all_dialogs if d.get("has_pd"))
        without_pd = len(all_dialogs) - with_pd

        dataset = {
            "metadata": {
                "total_dialogs": len(all_dialogs),
                "generation_timestamp": time.time(),
                "model_used": self.model,
                "generation_stats": self.stats
            },
            "dialogs": all_dialogs
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"✅[async] Датасет сохранен: {save_path}")

        return {
            "total_dialogs": len(all_dialogs),
            "with_pd": with_pd,
            "without_pd": without_pd,
            "successful_batches": self.stats["successful_batches"],
            "failed_batches": self.stats["failed_batches"],
            "total_cost_tokens": self.stats["total_cost_tokens"],
            "sphere_distribution": sphere_stats,
            "save_path": save_path
        }


def main():
    """Основная функция для тестирования генератора"""
    print("🚀 Тестирование OpenAI генератора датасета")
    
    # Для тестирования создаем небольшой датасет
    # Загружаем базовый конфиг, чтобы получить модель и URL
    try:
        from .config_loader import ConfigLoader
        loader = ConfigLoader()
        api_key = loader.get_str("OPENAI_API_KEY")
        model = loader.get_str("MODEL")
        base_url = loader.get_str("OPENAI_BASE_URL")
    except Exception:
        api_key, model, base_url = None, None, None

    generator = OpenAIDatasetGenerator(api_key=api_key, model=model, base_url=base_url)
    
    try:
        stats = generator.generate_dataset(
            total_dialogs=50,  # Небольшой тест
            max_batch_size=15,
            save_path="data/generated/test_openai_dataset.json"
        )
        
        print(f"\n🎉 Тестирование завершено!")
        print(f"📄 Файл: {stats['save_path']}")
        
        # Показываем пример диалога
        try:
            with open(stats['save_path'], 'r', encoding='utf-8') as f:
                dialogs = json.load(f)
                
            if dialogs:
                print(f"\n📝 ПРИМЕР ДИАЛОГА:")
                sample = dialogs[0]
                print(f"ID: {sample['id']}")
                print(f"Текст: {sample['text'][:150]}...")
                print(f"ПД: {sample['has_pd']}, сущностей: {len(sample['entities'])}")
                
        except Exception as e:
            print(f"❌ Ошибка чтения результата: {e}")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        print("💡 Убедитесь, что установлен OpenAI API ключ:")
        print("   export OPENAI_API_KEY='your-api-key-here'")


if __name__ == "__main__":
    main() 