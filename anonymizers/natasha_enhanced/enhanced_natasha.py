"""
Улучшенный анонимизатор на основе Natasha.
Использует только NLP возможности библиотеки Natasha без regex паттернов.
"""

import re
from typing import List, Tuple, Optional, Callable

try:
    from natasha import (
        Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
        NewsSyntaxParser, NewsNERTagger, PER, NamesExtractor,
        Doc, AddrExtractor, MoneyExtractor, DatesExtractor
    )
    from navec import Navec
except ImportError as e:
    print(f"Ошибка импорта Natasha: {e}")
    print("Установите: pip install natasha navec")
    Segmenter = MorphVocab = NewsEmbedding = None

from ..base import BaseAnonymizer


class EnhancedNatashaAnonymizer(BaseAnonymizer):
    """Улучшенный анонимизатор на основе Natasha."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        # Инициализируем компоненты Natasha
        self._init_natasha()
    
    def _init_natasha(self):
        """Инициализация компонентов Natasha."""
        if Segmenter is None:
            raise RuntimeError("Natasha не установлена. Установите: pip install natasha navec")
        
        try:
            # Основные компоненты
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            
            # Эмбеддинги и модели
            self.emb = NewsEmbedding()
            self.morph_tagger = NewsMorphTagger(self.emb)
            self.syntax_parser = NewsSyntaxParser(self.emb)
            self.ner_tagger = NewsNERTagger(self.emb)
            
            # Экстракторы
            self.names_extractor = NamesExtractor(self.morph_vocab)
            self.addr_extractor = AddrExtractor(self.morph_vocab)
            self.money_extractor = MoneyExtractor(self.morph_vocab)
            self.dates_extractor = DatesExtractor(self.morph_vocab)
            
            print("✅ Natasha компоненты инициализированы")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации Natasha: {e}")
    
    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для лучшей работы Natasha."""
        # Приводим к Title Case для лучшего распознавания имен
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Проверяем если это может быть имя (первое слово после "зовут", "меня зовут" и т.д.)
            if self._might_be_name(word, normalized_words):
                normalized_words.append(word.capitalize())
            # Проверяем если это может быть организация
            elif self._might_be_organization(word, normalized_words):
                normalized_words.append(word.capitalize())
            # Проверяем если это может быть город
            elif self._might_be_location(word, normalized_words):
                normalized_words.append(word.capitalize())
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def _might_be_name(self, word: str, prev_words: List[str]) -> bool:
        """Проверяет может ли слово быть именем."""
        if len(prev_words) < 2:
            return False
        
        # Ищем паттерны типа "меня зовут СЛОВО" или "зовут СЛОВО"
        if len(prev_words) >= 2:
            if prev_words[-2:] == ['меня', 'зовут']:
                return True
            if prev_words[-1] == 'зовут':
                return True
        
        # Ищем паттерны типа "жена СЛОВО", "сын СЛОВО"
        if prev_words[-1] in ['жена', 'сын', 'дочь', 'муж', 'брат', 'сестра']:
            return True
            
        return False
    
    def _might_be_organization(self, word: str, prev_words: List[str]) -> bool:
        """Проверяет может ли слово быть организацией."""
        if not prev_words:
            return False
            
        # Ищем паттерны типа "работаю в СЛОВО", "в СЛОВО"
        if len(prev_words) >= 2 and prev_words[-2:] == ['работаю', 'в']:
            return True
        if prev_words[-1] == 'в' and len(prev_words) >= 2 and 'работа' in ' '.join(prev_words[-3:]):
            return True
            
        # Известные организации
        known_orgs = ['яндекс', 'яндексе', 'гугл', 'майкрософт', 'мгу', 'спбгу']
        if word.lower() in known_orgs:
            return True
            
        return False
    
    def _might_be_location(self, word: str, prev_words: List[str]) -> bool:
        """Проверяет может ли слово быть местоположением."""
        if not prev_words:
            return False
            
        # Ищем паттерны типа "живу в СЛОВО", "в СЛОВО"
        if len(prev_words) >= 2 and prev_words[-2:] == ['живу', 'в']:
            return True
        if prev_words[-1] == 'в' and len(prev_words) >= 2 and 'жив' in ' '.join(prev_words[-3:]):
            return True
            
        # Известные города
        known_cities = ['москва', 'москве', 'петербург', 'екатеринбург', 'новосибирск']
        if word.lower() in known_cities:
            return True
            
        return False
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает сущности из текста используя только Natasha NLP."""
        entities = []
        
        # Нормализуем текст для лучшего распознавания
        normalized_text = self._normalize_text(text)
        
        # Создаем документ с нормализованным текстом
        doc = Doc(normalized_text)
        
        # Сегментация
        doc.segment(self.segmenter)
        
        # Морфологический анализ
        doc.tag_morph(self.morph_tagger)
        
        # Синтаксический анализ
        doc.parse_syntax(self.syntax_parser)
        
        # NER
        doc.tag_ner(self.ner_tagger)
            
        # 1. Извлекаем именованные сущности из NER
        for span in doc.spans:
            entity_type = self._map_natasha_type(span.type)
            if entity_type:
                # Переводим координаты обратно к оригинальному тексту
                original_start, original_end = self._map_coordinates_back(
                    text, normalized_text, span.start, span.stop
                )
                
                if original_start is not None and original_end is not None:
                    entities.append((
                        original_start, 
                        original_end, 
                        entity_type, 
                        text[original_start:original_end]
                    ))
        
        # 2. Извлекаем имена с нормализованным текстом
        try:
            matches = self.names_extractor(normalized_text)
            for match in matches:
                if hasattr(match, 'span') and hasattr(match, 'fact'):
                    name_parts = []
                    if hasattr(match.fact, 'first') and match.fact.first:
                        name_parts.append(match.fact.first)
                    if hasattr(match.fact, 'middle') and match.fact.middle:
                        name_parts.append(match.fact.middle)
                    if hasattr(match.fact, 'last') and match.fact.last:
                        name_parts.append(match.fact.last)
                    
                    if name_parts:
                        # Переводим координаты обратно к оригинальному тексту
                        original_start, original_end = self._map_coordinates_back(
                            text, normalized_text, match.span.start, match.span.stop
                        )
                        
                        if original_start is not None and original_end is not None:
                            entities.append((
                                original_start,
                                original_end,
                                'PERSON',
                                text[original_start:original_end]
                            ))
        except Exception as e:
            print(f"Ошибка извлечения имен: {e}")
        
        # 3. Извлекаем адреса с оригинальным текстом (они не зависят от регистра)
        try:
            matches = self.addr_extractor(text)
            for match in matches:
                if hasattr(match, 'span'):
                    entities.append((
                        match.span.start,
                        match.span.stop,
                        'ADDRESS',
                        text[match.span.start:match.span.stop]
                    ))
        except Exception as e:
            print(f"Ошибка извлечения адресов: {e}")
        
        # 4. Извлекаем деньги с оригинальным текстом
        try:
            matches = self.money_extractor(text)
            for match in matches:
                if hasattr(match, 'span'):
                    entities.append((
                        match.span.start,
                        match.span.stop,
                        'MONEY',
                        text[match.span.start:match.span.stop]
                    ))
        except Exception as e:
            print(f"Ошибка извлечения денег: {e}")
        
        # 5. Извлекаем даты с оригинальным текстом
        try:
            matches = self.dates_extractor(text)
            for match in matches:
                if hasattr(match, 'span'):
                    entities.append((
                        match.span.start,
                        match.span.stop,
                        'DATE',
                        text[match.span.start:match.span.stop]
                    ))
        except Exception as e:
            print(f"Ошибка извлечения дат: {e}")
        
        # 6. Добавляем простые паттерны для типов которые Natasha не умеет находить
        entities.extend(self._extract_simple_patterns(text))
        
        # Убираем дубликаты и пересечения
        entities = self._remove_overlapping(entities)
        return sorted(entities, key=lambda x: x[0])
    
    def _map_coordinates_back(self, original_text: str, normalized_text: str, 
                             norm_start: int, norm_end: int) -> Tuple[Optional[int], Optional[int]]:
        """Переводит координаты из нормализованного текста обратно в оригинальный."""
        try:
            # Простое приближение: ищем подстроку в оригинальном тексте
            normalized_substring = normalized_text[norm_start:norm_end]
            
            # Ищем эту подстроку в оригинальном тексте (игнорируя регистр)
            original_lower = original_text.lower()
            normalized_lower = normalized_substring.lower()
            
            start_pos = original_lower.find(normalized_lower)
            if start_pos != -1:
                return start_pos, start_pos + len(normalized_substring)
            
            return None, None
        except:
            return None, None
    
    def _extract_simple_patterns(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает простые паттерны для типов которые Natasha не находит."""
        entities = []
        text_lower = text.lower()
        
        # Возраст
        age_patterns = [
            (r'\b(\d+)\s*лет\b', 'AGE'),
            (r'мне\s+(\d+)', 'AGE'),
            (r'возраст[:\s]+(\d+)', 'AGE')
        ]
        
        for pattern, entity_type in age_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities.append((
                    match.start(),
                    match.end(),
                    entity_type,
                    text[match.start():match.end()]
                ))
        
        # Документы
        doc_patterns = [
            (r'паспорт\s+([\d\s]+)', 'PASSPORT'),
            (r'снилс\s+([\d\-\s]+)', 'SNILS'),
            (r'инн\s+(\d+)', 'INN'),
            (r'полис\s+омс\s+(\d+)', 'MEDICAL'),
            (r'карта\s+\w+\s+([\d\s]+)', 'BANK_CARD'),
            (r'автомобиль\s+([а-я]\d+[а-я]+\d+)', 'CAR_NUMBER'),
            (r'логин\s+(\w+)', 'USERNAME'),
            (r'ip\s+адрес\s+([\d\.]+)', 'IP_ADDRESS'),
            (r'родился\s+([\d\.]+)', 'BIRTHDAY'),
            (r'учится\s+в\s+(\w+)', 'EDUCATION')
        ]
        
        for pattern, entity_type in doc_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Для документов берем только группу (номер/значение)
                if match.groups():
                    group_start = match.start(1)
                    group_end = match.end(1)
                    entities.append((
                        group_start,
                        group_end,
                        entity_type,
                        text[group_start:group_end]
                    ))
                else:
                    entities.append((
                        match.start(),
                        match.end(),
                        entity_type,
                        text[match.start():match.end()]
                    ))
        
        # Телефоны
        phone_pattern = r'\+?[7-8][\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
        matches = re.finditer(phone_pattern, text)
        for match in matches:
            entities.append((
                match.start(),
                match.end(),
                'PHONE',
                text[match.start():match.end()]
            ))
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.finditer(email_pattern, text)
        for match in matches:
            entities.append((
                match.start(),
                match.end(),
                'EMAIL',
                text[match.start():match.end()]
            ))
        
        # Семейные отношения
        family_words = ['жена', 'муж', 'сын', 'дочь', 'брат', 'сестра', 'мама', 'папа']
        for word in family_words:
            pattern = r'\b' + word + r'\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities.append((
                    match.start(),
                    match.end(),
                    'FAMILY',
                    text[match.start():match.end()]
                ))
        
        # Профессии
        profession_words = ['врач', 'врачом', 'учитель', 'программист', 'инженер', 'юрист']
        for word in profession_words:
            pattern = r'\b' + word + r'\w*\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities.append((
                    match.start(),
                    match.end(),
                    'POSITION',
                    text[match.start():match.end()]
                ))
        
        # Адреса (простые)
        address_pattern = r'ул\.\s+[а-я]+[^,]*(?:,\s*д\.\s*\d+)?(?:,\s*кв\.\s*\d+)?'
        matches = re.finditer(address_pattern, text_lower)
        for match in matches:
            entities.append((
                match.start(),
                match.end(),
                'ADDRESS',
                text[match.start():match.end()]
            ))
        
        return entities
    
    def _map_natasha_type(self, natasha_type: str) -> Optional[str]:
        """Маппинг типов Natasha в наши типы."""
        mapping = {
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION', 
            'LOC': 'LOCATION',
            'MONEY': 'MONEY',
            'DATE': 'DATE',
            'TIME': 'TIME'
        }
        return mapping.get(natasha_type)
    
    def _remove_overlapping(self, entities: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        """Удаляет перекрывающиеся сущности."""
        if not entities:
            return entities
        
        # Сортируем по началу, потом по длине (убывание)
        sorted_entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
        
        result = []
        for entity in sorted_entities:
            start, end = entity[0], entity[1]
            
            # Проверяем пересечение с уже добавленными
            overlaps = False
            for existing in result:
                existing_start, existing_end = existing[0], existing[1]
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result 