"""
Базовый класс для всех анонимизаторов.
Определяет общий интерфейс и функции анонимизации.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Optional
import random
import re
from faker import Faker


class BaseAnonymizer(ABC):
    """Базовый класс для всех анонимизаторов."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None
    ):
        """
        Args:
            aggressiveness: Уровень агрессивности (0.0 - консервативный, 1.0 - агрессивный)
            anonymization_func: Функция анонимизации (entity_type, text) -> anonymized_text
        """
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))
        self.anonymization_func = anonymization_func or self._default_anonymization_func
        self.fake = Faker('ru_RU')
        
    def _default_anonymization_func(self, entity_type: str, text: str) -> str:
        """Стандартная функция анонимизации с заглушками."""
        return f"<{entity_type}>"
    
    def _faker_anonymization_func(self, entity_type: str, text: str) -> str:
        """Функция анонимизации с генерацией похожих данных через Faker."""
        faker_map = {
            'PERSON': lambda: self.fake.name(),
            'EMAIL': lambda: self.fake.email(),
            'PHONE': lambda: self.fake.phone_number(),
            'ADDRESS': lambda: self.fake.address().replace('\n', ', '),
            'ORGANIZATION': lambda: self.fake.company(),
            'PASSPORT': lambda: f"{random.randint(10, 99)} {random.randint(10, 99)} {random.randint(100000, 999999)}",
            'SNILS': lambda: f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(100, 999)} {random.randint(10, 99)}",
            'INN': lambda: ''.join([str(random.randint(0, 9)) for _ in range(12)]),
            'CARD': lambda: f"{random.randint(1000, 9999)} **** **** {random.randint(1000, 9999)}",
            'BIRTHDAY': lambda: self.fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%d.%m.%Y'),
            'AGE': lambda: str(random.randint(18, 80)),
            'LOCATION': lambda: self.fake.city(),
            'EDUCATION': lambda: random.choice(['МГУ', 'СПбГУ', 'МФТИ', 'ВШЭ', 'МГТУ']),
            'POSITION': lambda: random.choice(['менеджер', 'специалист', 'директор', 'инженер']),
            'FAMILY': lambda: random.choice(['супруг', 'супруга', 'сын', 'дочь', 'родитель']),
            'IP_ADDRESS': lambda: self.fake.ipv4(),
            'URL': lambda: self.fake.url(),
            'USERNAME': lambda: self.fake.user_name(),
            'CAR_NUMBER': lambda: f"{random.choice(['А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'])}{random.randint(100, 999)}{random.choice(['А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'])}{random.choice(['А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'])}",
        }
        
        if entity_type in faker_map:
            return faker_map[entity_type]()
        return f"<{entity_type}>"
    
    def set_faker_mode(self):
        """Переключиться на режим генерации похожих данных."""
        self.anonymization_func = self._faker_anonymization_func
    
    def set_placeholder_mode(self):
        """Переключиться на режим заглушек."""
        self.anonymization_func = self._default_anonymization_func
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Извлекает сущности из текста.
        
        Returns:
            List[Tuple[start, end, entity_type, entity_text]]
        """
        pass
    
    def anonymize(self, text: str) -> str:
        """
        Анонимизирует текст.
        
        Args:
            text: Исходный текст
            
        Returns:
            Анонимизированный текст
        """
        entities = self.extract_entities(text)
        
        # Сортируем по позиции с конца, чтобы не сбить индексы
        entities.sort(key=lambda x: x[0], reverse=True)
        
        result = text
        for start, end, entity_type, entity_text in entities:
            # Применяем порог агрессивности
            if self._should_anonymize(entity_type, entity_text):
                replacement = self.anonymization_func(entity_type, entity_text)
                result = result[:start] + replacement + result[end:]
        
        return result
    
    def _should_anonymize(self, entity_type: str, entity_text: str) -> bool:
        """Определяет, нужно ли анонимизировать данную сущность."""
        # Консервативные типы - анонимизируем только при высокой уверенности
        conservative_types = {'PERSON', 'PHONE', 'EMAIL', 'PASSPORT', 'SNILS', 'INN'}
        
        if entity_type in conservative_types:
            return self.aggressiveness > 0.3
        
        # Агрессивные типы - анонимизируем при низкой уверенности
        return self.aggressiveness > 0.1
    
    def analyze(self, text: str) -> List[Dict[str, any]]:
        """
        Анализирует текст и возвращает найденные сущности.
        
        Returns:
            List[Dict] с информацией о найденных сущностях
        """
        entities = self.extract_entities(text)
        
        result = []
        for start, end, entity_type, entity_text in entities:
            result.append({
                'start': start,
                'end': end,
                'type': entity_type,
                'text': entity_text,
                'length': end - start,
                'will_anonymize': self._should_anonymize(entity_type, entity_text)
            })
        
        return result 