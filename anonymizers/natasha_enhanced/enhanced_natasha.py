"""
Расширенный анонимизатор на основе библиотеки Natasha.
Использует комбинацию NER и rule-based подходов для русского языка.
"""

from typing import List, Tuple, Optional, Callable

try:
    from natasha import (
        Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
        NewsSyntaxParser, NewsNERTagger, PER, NamesExtractor,
        Doc, AddrExtractor, DatesExtractor
    )
    from navec import Navec
except ImportError as e:
    print(f"Ошибка импорта Natasha: {e}")
    print("Установите: pip install natasha navec")
    Segmenter = MorphVocab = NewsEmbedding = None

from ..base import BaseAnonymizer


class EnhancedNatashaAnonymizer(BaseAnonymizer):
    """Расширенный анонимизатор на основе Natasha с дополнительными правилами."""
    
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
            self.dates_extractor = DatesExtractor(self.morph_vocab)
            
            print("✅ Natasha компоненты инициализированы")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации Natasha: {e}")
    
    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для лучшего распознавания."""
        # Убираем лишние пробелы и переносы строк
        text = ' '.join(text.split())
        
        # Убираем множественные знаки препинания
        import string
        for punct in string.punctuation:
            if punct in '.,!?;:':
                text = text.replace(punct + punct, punct)
        
        return text
    
    def _might_be_name(self, word: str, prev_words: List[str]) -> bool:
        """Эвристика для определения возможного имени."""
        if len(word) < 2:
            return False
        
        # Проверяем заглавную букву
        if not word[0].isupper():
            return False
        
        # Контекстные подсказки
        name_contexts = ['зовут', 'меня', 'имя', 'фамилия']
        if any(ctx in ' '.join(prev_words[-3:]).lower() for ctx in name_contexts):
            return True
            
        return False
    
    def _might_be_organization(self, word: str, prev_words: List[str]) -> bool:
        """Эвристика для определения возможной организации."""
        org_suffixes = ['ооо', 'зао', 'оао', 'ип', 'ltd', 'inc']
        if word.lower() in org_suffixes:
            return True
            
        org_contexts = ['работаю', 'компания', 'фирма', 'предприятие']
        if any(ctx in ' '.join(prev_words[-3:]).lower() for ctx in org_contexts):
            return True
            
        return False
    
    def _might_be_location(self, word: str, prev_words: List[str]) -> bool:
        """Эвристика для определения возможного места."""
        location_contexts = ['живу', 'город', 'село', 'деревня', 'улица', 'проспект']
        if any(ctx in ' '.join(prev_words[-3:]).lower() for ctx in location_contexts):
            return True
            
        return False
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Извлекает именованные сущности из текста используя только Natasha NER.
        
        Args:
            text: Исходный текст
            
        Returns:
            Список кортежей (start, end, type, value)
        """
        entities = []
        
        # Нормализуем текст для лучшего распознавания
        normalized_text = self._normalize_text(text)
        
        # 1. Основной NER с Natasha
        try:
            doc = Doc(normalized_text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.ner_tagger)
            
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
        except Exception as e:
            print(f"Ошибка основного NER: {e}")
        
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
                
        # 4. Извлекаем даты с оригинальным текстом
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
    
    def _map_natasha_type(self, natasha_type: str) -> Optional[str]:
        """Маппинг типов Natasha в наши типы."""
        mapping = {
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION', 
            'LOC': 'LOCATION',
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