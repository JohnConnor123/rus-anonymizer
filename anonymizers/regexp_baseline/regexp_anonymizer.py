"""
Чистый RegExp Baseline анонимизатор.
Использует только регулярные выражения без ML компонентов.
"""

import re
from typing import List, Tuple, Optional, Callable

from ..base import BaseAnonymizer


class RegExpBaselineAnonymizer(BaseAnonymizer):
    """
    Baseline анонимизатор на основе только регулярных выражений.
    Служит для сравнения с ML-подходами.
    """
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        # Компилируем все регулярные выражения для производительности
        self._compile_patterns()
        
        print("✅ RegExp Baseline анонимизатор инициализирован")
    
    def _compile_patterns(self):
        """Компилирует все регулярные выражения."""
        
        # Телефоны (российские)
        self.phone_patterns = [
            re.compile(r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),  # +7 (XXX) XXX-XX-XX
            re.compile(r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),      # +7 XXX XXX-XX-XX
            re.compile(r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),    # 8 (XXX) XXX-XX-XX
            re.compile(r'8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),        # 8 XXX XXX-XX-XX
            re.compile(r'\+7\d{10}'),                                        # +7XXXXXXXXXX
            re.compile(r'8\d{10}'),                                          # 8XXXXXXXXXX
        ]
        
        # Email
        self.email_patterns = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            re.compile(r'email:\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'),
        ]
        
        # Паспорт РФ
        self.passport_patterns = [
            re.compile(r'\b\d{2}\s+\d{2}\s+\d{6}\b'),                       # XX XX XXXXXX
            re.compile(r'\b\d{4}\s+\d{6}\b'),                               # XXXX XXXXXX
            re.compile(r'паспорт[:\s]*\d{2}\s+\d{2}\s+\d{6}', re.IGNORECASE),
        ]
        
        # СНИЛС
        self.snils_patterns = [
            re.compile(r'\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b'),     # XXX-XXX-XXX XX
            re.compile(r'снилс[:\s]*\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}', re.IGNORECASE),
        ]
        
        # ИНН
        self.inn_patterns = [
            re.compile(r'\b\d{10}\b'),                                       # 10 цифр (юр.лицо)
            re.compile(r'\b\d{12}\b'),                                       # 12 цифр (физ.лицо)
            re.compile(r'инн[:\s]*\d{10,12}', re.IGNORECASE),
        ]
        
        # Банковские карты
        self.card_patterns = [
            re.compile(r'\b\d{4}[\s-]*\d{4}[\s-]*\d{4}[\s-]*\d{4}\b'),     # XXXX-XXXX-XXXX-XXXX
            re.compile(r'\b\d{4}[\s*]*\*{4}[\s*]*\*{4}[\s*]*\d{4}\b'),     # XXXX ****-****-XXXX
        ]
        
        # Даты рождения и даты
        self.birthday_patterns = [
            re.compile(r'\b\d{1,2}\.\d{1,2}\.\d{4}\b'),                     # DD.MM.YYYY
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),                       # DD/MM/YYYY
            re.compile(r'\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}', re.IGNORECASE),
            re.compile(r'родил[ас]ь[:\s]*\d{1,2}\.\d{1,2}\.\d{4}', re.IGNORECASE),
        ]
        
        # IP адреса
        self.ip_patterns = [
            re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        ]
        
        # URL
        self.url_patterns = [
            re.compile(r'https?://[^\s]+'),
            re.compile(r'www\.[^\s]+'),
        ]
        
        # Сокращенные имена (И.О., И. О.)
        self.short_name_patterns = [
            re.compile(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.(?:\s*[А-ЯЁ]\.)?', re.IGNORECASE),  # Иванов И.О.
            re.compile(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.', re.IGNORECASE),      # Иванов И. О.
        ]
        
        # Русские имена (простые паттерны)
        self.russian_name_patterns = [
            re.compile(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b'),  # Имя Фамилия [Отчество]
        ]
        
        # Номера автомобилей
        self.car_number_patterns = [
            re.compile(r'\b[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}\b'),
        ]
        
        # Возраст
        self.age_patterns = [
            re.compile(r'мне\s+\d{1,2}\s+лет', re.IGNORECASE),
            re.compile(r'\b\d{1,2}\s+лет\b', re.IGNORECASE),
        ]
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает сущности используя только регулярные выражения."""
        entities = []
        
        # Телефоны
        for pattern in self.phone_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'PHONE', match.group()))
        
        # Email
        for pattern in self.email_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'EMAIL', match.group()))
        
        # Паспорта
        for pattern in self.passport_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'PASSPORT', match.group()))
        
        # СНИЛС
        for pattern in self.snils_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'SNILS', match.group()))
        
        # ИНН
        for pattern in self.inn_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'INN', match.group()))
        
        # Банковские карты
        for pattern in self.card_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'BANK_CARD', match.group()))
        
        # Даты рождения
        for pattern in self.birthday_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'BIRTHDAY', match.group()))
        
        # IP адреса
        for pattern in self.ip_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'IP_ADDRESS', match.group()))
        
        # URL
        for pattern in self.url_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'URL', match.group()))
        
        # Сокращенные имена
        for pattern in self.short_name_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'PERSON', match.group()))
        
        # Русские имена (только если агрессивность высокая)
        if self.aggressiveness > 0.6:
            for pattern in self.russian_name_patterns:
                for match in pattern.finditer(text):
                    # Проверяем, что это не пересекается с уже найденными сущностями
                    if not self._overlaps_with_existing(match.start(), match.end(), entities):
                        entities.append((match.start(), match.end(), 'PERSON', match.group()))
        
        # Номера автомобилей
        for pattern in self.car_number_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'CAR_NUMBER', match.group()))
        
        # Возраст
        for pattern in self.age_patterns:
            for match in pattern.finditer(text):
                entities.append((match.start(), match.end(), 'AGE', match.group()))
        
        # Удаляем дубликаты и пересечения
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _overlaps_with_existing(self, start: int, end: int, entities: List[Tuple[int, int, str, str]]) -> bool:
        """Проверяет, пересекается ли новая сущность с уже найденными."""
        for entity_start, entity_end, _, _ in entities:
            if not (end <= entity_start or start >= entity_end):
                return True
        return False
    
    def _remove_overlaps(self, entities: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        """Удаляет пересекающиеся сущности, оставляя более специфичные."""
        if not entities:
            return entities
        
        # Сортируем по позиции
        entities.sort(key=lambda x: (x[0], x[1]))
        
        # Приоритеты типов сущностей (чем выше число, тем выше приоритет)
        priority = {
            'PHONE': 10,
            'EMAIL': 10,
            'PASSPORT': 9,
            'SNILS': 9,
            'INN': 8,
            'BANK_CARD': 8,
            'BIRTHDAY': 7,
            'PERSON': 7,
            'IP_ADDRESS': 5,
            'URL': 5,
            'CAR_NUMBER': 4,
            'AGE': 3,
        }
        
        result = []
        for entity in entities:
            start, end, entity_type, text = entity
            
            # Проверяем пересечения с уже добавленными сущностями
            overlaps = False
            for i, (res_start, res_end, res_type, res_text) in enumerate(result):
                if not (end <= res_start or start >= res_end):  # Есть пересечение
                    # Сравниваем приоритеты
                    current_priority = priority.get(entity_type, 1)
                    existing_priority = priority.get(res_type, 1)
                    
                    if current_priority > existing_priority:
                        # Заменяем существующую сущность
                        result[i] = entity
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result 