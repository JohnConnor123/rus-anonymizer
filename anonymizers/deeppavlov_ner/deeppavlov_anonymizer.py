"""
Анонимизатор на основе DeepPavlov NER.
Использует только возможности DeepPavlov без regex паттернов.
"""

from typing import List, Tuple, Optional, Callable, Dict, Any

try:
    from deeppavlov import build_model, configs
    from deeppavlov.core.commands.utils import parse_config
except ImportError as e:
    print(f"Ошибка импорта DeepPavlov: {e}")
    print("Установите: pip install deeppavlov")
    build_model = configs = parse_config = None

from ..base import BaseAnonymizer


class DeepPavlovAnonymizer(BaseAnonymizer):
    """Анонимизатор на основе DeepPavlov NER."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None,
        model_name: str = "ner_ontonotes_bert_mult"
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        self.model_name = model_name
        self.ner_model = None
        
        # Инициализируем DeepPavlov модель
        self._init_model()
    
    def _init_model(self):
        """Инициализация DeepPavlov модели."""
        if build_model is None:
            raise RuntimeError("DeepPavlov не установлен. Установите: pip install deeppavlov")
        
        try:
            print(f"Загружаю DeepPavlov модель: {self.model_name}")
            
            # Загружаем конфигурацию модели
            config = getattr(configs.ner, self.model_name)
            
            # Строим модель
            self.ner_model = build_model(config, download=True)
            
            print("✅ DeepPavlov NER модель загружена")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации DeepPavlov: {e}")
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает сущности из текста используя только DeepPavlov."""
        if self.ner_model is None:
            raise RuntimeError("DeepPavlov модель не инициализирована")
        
        entities = []
        
        try:
            # Получаем предсказания от DeepPavlov
            tokens, tags = self.ner_model([text])
            
            if tokens and tags:
                entities = self._parse_bio_tags(text, tokens[0], tags[0])
            
        except Exception as e:
            print(f"Ошибка DeepPavlov NER: {e}")
            return []
        
        # Убираем дубликаты и пересечения
        entities = self._remove_overlapping(entities)
        return sorted(entities, key=lambda x: x[0])
    
    def _parse_bio_tags(self, text: str, tokens: List[str], tags: List[str]) -> List[Tuple[int, int, str, str]]:
        """Парсит BIO теги в сущности."""
        entities = []
        current_entity = None
        current_start = 0
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith('B-'):
                # Начало новой сущности
                if current_entity:
                    # Завершаем предыдущую сущность
                    entities.append(current_entity)
                
                entity_type = self._map_deeppavlov_type(tag[2:])
                if entity_type:
                    # Находим позицию токена в тексте
                    token_start = text.find(token, current_start)
                    if token_start != -1:
                        current_entity = [token_start, token_start + len(token), entity_type, token]
                        current_start = token_start + len(token)
                    
            elif tag.startswith('I-') and current_entity:
                # Продолжение сущности
                entity_type = self._map_deeppavlov_type(tag[2:])
                if entity_type == current_entity[2]:
                    # Расширяем текущую сущность
                    token_start = text.find(token, current_start)
                    if token_start != -1:
                        current_entity[1] = token_start + len(token)
                        current_entity[3] += " " + token
                        current_start = token_start + len(token)
                        
            else:
                # O тег или конец сущности
                if current_entity:
                    entities.append(tuple(current_entity))
                    current_entity = None
                
                # Обновляем позицию для поиска следующих токенов
                token_pos = text.find(token, current_start)
                if token_pos != -1:
                    current_start = token_pos + len(token)
        
        # Добавляем последнюю сущность если есть
        if current_entity:
            entities.append(tuple(current_entity))
        
        return entities
    
    def _map_deeppavlov_type(self, dp_type: str) -> Optional[str]:
        """Маппинг типов DeepPavlov в наши типы."""
        mapping = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'LOCATION',
            'MISC': 'MISC',
            'DATE': 'BIRTHDAY',
        }
        
        return mapping.get(dp_type.upper())
    
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


class DeepPavlovMultiModelAnonymizer(BaseAnonymizer):
    """Анонимизатор с несколькими моделями DeepPavlov."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """Инициализация нескольких моделей DeepPavlov."""
        if build_model is None:
            raise RuntimeError("DeepPavlov не установлен. Установите: pip install deeppavlov")
        
        model_configs = [
            "ner_ontonotes_bert_mult",
            "ner_conll2003_bert",
        ]
        
        for model_name in model_configs:
            try:
                print(f"Загружаю модель: {model_name}")
                config = getattr(configs.ner, model_name)
                model = build_model(config, download=True)
                self.models[model_name] = model
                print(f"✅ {model_name} загружена")
            except Exception as e:
                print(f"❌ Ошибка загрузки {model_name}: {e}")
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает сущности используя все доступные модели."""
        all_entities = []
        
        for model_name, model in self.models.items():
            try:
                tokens, tags = model([text])
                if tokens and tags:
                    entities = self._parse_bio_tags(text, tokens[0], tags[0])
                    all_entities.extend(entities)
            except Exception as e:
                print(f"Ошибка модели {model_name}: {e}")
                continue
        
        # Убираем дубликаты и пересечения
        all_entities = self._remove_overlapping(all_entities)
        return sorted(all_entities, key=lambda x: x[0])
    
    def _parse_bio_tags(self, text: str, tokens: List[str], tags: List[str]) -> List[Tuple[int, int, str, str]]:
        """Парсит BIO теги в сущности."""
        entities = []
        current_entity = None
        current_start = 0
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = self._map_deeppavlov_type(tag[2:])
                if entity_type:
                    token_start = text.find(token, current_start)
                    if token_start != -1:
                        current_entity = [token_start, token_start + len(token), entity_type, token]
                        current_start = token_start + len(token)
                    
            elif tag.startswith('I-') and current_entity:
                entity_type = self._map_deeppavlov_type(tag[2:])
                if entity_type == current_entity[2]:
                    token_start = text.find(token, current_start)
                    if token_start != -1:
                        current_entity[1] = token_start + len(token)
                        current_entity[3] += " " + token
                        current_start = token_start + len(token)
                        
            else:
                if current_entity:
                    entities.append(tuple(current_entity))
                    current_entity = None
                
                token_pos = text.find(token, current_start)
                if token_pos != -1:
                    current_start = token_pos + len(token)
        
        if current_entity:
            entities.append(tuple(current_entity))
        
        return entities
    
    def _map_deeppavlov_type(self, dp_type: str) -> Optional[str]:
        """Маппинг типов DeepPavlov в наши типы."""
        mapping = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'LOCATION',
            'MISC': 'MISC',
            'DATE': 'BIRTHDAY',
        }
        
        return mapping.get(dp_type.upper())
    
    def _remove_overlapping(self, entities: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        """Удаляет перекрывающиеся сущности."""
        if not entities:
            return entities
        
        sorted_entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
        
        result = []
        for entity in sorted_entities:
            start, end = entity[0], entity[1]
            
            overlaps = False
            for existing in result:
                existing_start, existing_end = existing[0], existing[1]
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result 