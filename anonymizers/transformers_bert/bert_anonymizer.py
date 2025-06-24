"""
Анонимизатор на основе BERT трансформера.
Использует только возможности трансформера без regex паттернов.
"""

from typing import List, Tuple, Optional, Callable, Dict, Any

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
except ImportError as e:
    print(f"Ошибка импорта transformers: {e}")
    print("Установите: pip install transformers torch")
    AutoTokenizer = AutoModelForTokenClassification = pipeline = torch = None

from ..base import BaseAnonymizer


class BertAnonymizer(BaseAnonymizer):
    """Анонимизатор на основе BERT трансформера."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None,
        model_name: str = "Gherman/bert-base-NER-Russian"  # Русская NER модель
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        
        # Инициализируем BERT модель
        self._init_model()
    
    def _init_model(self):
        """Инициализация BERT модели."""
        if AutoTokenizer is None:
            raise RuntimeError("Transformers не установлены. Установите: pip install transformers torch")
        
        try:
            print(f"Загружаю BERT модель: {self.model_name}")
            
            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            # Создаем NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("✅ BERT NER pipeline создан успешно")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации BERT: {e}")
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает сущности из текста используя только BERT."""
        if self.ner_pipeline is None:
            raise RuntimeError("BERT модель не инициализирована")
        
        entities = []
        
        try:
            # Получаем предсказания от BERT
            predictions = self.ner_pipeline(text)
            
            for pred in predictions:
                entity_type = self._map_bert_type(pred['entity_group'])
                if entity_type and pred['score'] > 0.5:  # Порог уверенности
                    entities.append((
                        pred['start'],
                        pred['end'],
                        entity_type,
                        pred['word']
                    ))
            
        except Exception as e:
            print(f"Ошибка BERT NER: {e}")
            return []
        
        # Убираем дубликаты и пересечения
        entities = self._remove_overlapping(entities)
        return sorted(entities, key=lambda x: x[0])
    
    def _map_bert_type(self, bert_type: str) -> Optional[str]:
        """Маппинг типов BERT в наши типы."""
        # Убираем префиксы B- и I-
        clean_type = bert_type.replace('B-', '').replace('I-', '')
        
        mapping = {
            # Стандартные типы
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'MISC': 'MISC',
            'GPE': 'LOCATION',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'PERCENT': 'PERCENT',
            
            # Добавляем новые типы от русской модели
            'FIRST_NAME': 'PERSON',
            'LAST_NAME': 'PERSON', 
            'MIDDLE_NAME': 'PERSON',
            'CITY': 'LOCATION',
            'COUNTRY': 'LOCATION',
            'NATIONALITY': 'PERSON',
            'JOB': 'POSITION',
            'AGE': 'AGE',
            'ORG_NAME': 'ORGANIZATION',
            'FACILITY': 'LOCATION',
            'STREET': 'ADDRESS',
            'HOUSE': 'ADDRESS',
            'PHONE': 'PHONE',
            'EMAIL': 'EMAIL'
        }
        
        return mapping.get(clean_type.upper())
    
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


class RuBertAnonymizer(BertAnonymizer):
    """Специализированный анонимизатор для русского языка на основе RuBERT."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None
    ):
        # Используем лучшую русскую модель
        super().__init__(
            aggressiveness, 
            anonymization_func, 
            "DeepPavlov/rubert-base-cased"
        )

class MultiBertAnonymizer(BaseAnonymizer):
    """Анонимизатор, использующий несколько BERT моделей одновременно."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        # Список моделей для использования
        self.model_names = [
            "DeepPavlov/rubert-base-cased",
            "sberbank-ai/ruBert-base",
            "bert-base-multilingual-cased",
        ]
        
        self.anonymizers = []
        self._init_models()
    
    def _init_models(self):
        """Инициализация нескольких моделей."""
        print("Инициализация множественных BERT моделей...")
        
        for model_name in self.model_names:
            try:
                anonymizer = BertAnonymizer(
                    aggressiveness=self.aggressiveness,
                    anonymization_func=self.anonymization_func,
                    model_name=model_name
                )
                
                if anonymizer.ner_pipeline:  # Если модель загрузилась успешно
                    self.anonymizers.append(anonymizer)
                    print(f"Модель {model_name} добавлена")
                else:
                    print(f"Модель {model_name} не загрузилась")
                    
            except Exception as e:
                print(f"Ошибка загрузки модели {model_name}: {e}")
        
        print(f"Успешно загружено {len(self.anonymizers)} моделей")
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает сущности, используя все доступные модели."""
        all_entities = []
        
        # Собираем результаты от всех моделей
        for i, anonymizer in enumerate(self.anonymizers):
            try:
                entities = anonymizer.extract_entities(text)
                # Добавляем суффикс модели для отслеживания
                tagged_entities = [
                    (start, end, f"{entity_type}_M{i}", entity_text)
                    for start, end, entity_type, entity_text in entities
                ]
                all_entities.extend(tagged_entities)
                
            except Exception as e:
                print(f"Ошибка модели {i}: {e}")
        
        # Убираем дубликаты и пересечения
        all_entities = self._remove_overlapping(all_entities)
        
        # Убираем суффиксы моделей
        cleaned_entities = [
            (start, end, entity_type.split('_M')[0], entity_text)
            for start, end, entity_type, entity_text in all_entities
        ]
        
        return sorted(cleaned_entities, key=lambda x: x[0])
    
    def _remove_overlapping(self, entities: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        """Удаляет перекрывающиеся сущности с приоритетом по модели."""
        if not entities:
            return entities
        
        # Сортируем по началу, потом по длине (убывание), потом по приоритету модели
        def sort_key(entity):
            start, end, entity_type, text = entity
            # Приоритет: чем меньше номер модели, тем выше приоритет
            model_priority = 0
            if '_M' in entity_type:
                try:
                    model_priority = int(entity_type.split('_M')[1])
                except:
                    pass
            return (start, -(end - start), model_priority)
        
        sorted_entities = sorted(entities, key=sort_key)
        
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