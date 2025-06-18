"""
Анонимизатор на основе spaCy с расширенными возможностями.
Поддерживает EntityRuler для создания собственных правил распознавания.
"""

import re
from typing import List, Dict, Optional, Tuple, Callable, Any

try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.tokens import Doc, Span
    from spacy.lang.ru import Russian
except ImportError as e:
    print(f"Ошибка импорта spaCy: {e}")
    print("Установите: pip install spacy && python -m spacy download ru_core_news_sm")
    spacy = None

from ..base import BaseAnonymizer


class SpacyAnonymizer(BaseAnonymizer):
    """Анонимизатор на основе spaCy с расширенными правилами."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None,
        model_name: str = "ru_core_news_sm"
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        
        # Инициализируем spaCy модель
        self._init_model()
        
        # Добавляем собственные правила
        self._init_patterns()
    
    def _init_model(self):
        """Инициализация spaCy модели."""
        if spacy is None:
            raise RuntimeError("spaCy не установлен. Установите: pip install spacy && python -m spacy download ru_core_news_sm")
        
        try:
            # Пытаемся загрузить полную русскую модель
            self.nlp = spacy.load(self.model_name)
            print(f"✅ Загружена полная модель spaCy: {self.model_name}")
            
        except OSError:
            # Если полная модель недоступна, создаем пустую русскую модель
            print(f"⚠️  Модель {self.model_name} недоступна, создаю пустую русскую модель")
            
            # Создаем пустую русскую модель
            self.nlp = Russian()
            
            # Добавляем необходимые компоненты для работы с POS атрибутами
            if 'morphologizer' not in self.nlp.pipe_names:
                # Добавляем morphologizer для русского языка
                try:
                    morphologizer = self.nlp.add_pipe('morphologizer')
                    print("✅ Добавлен morphologizer")
                except Exception as e:
                    print(f"⚠️  Не удалось добавить morphologizer: {e}")
            
            if 'tagger' not in self.nlp.pipe_names:
                # Добавляем tagger как альтернативу
                try:
                    tagger = self.nlp.add_pipe('tagger')
                    print("✅ Добавлен tagger")
                except Exception as e:
                    print(f"⚠️  Не удалось добавить tagger: {e}")
            
            if 'attribute_ruler' not in self.nlp.pipe_names:
                # Добавляем attribute_ruler
                try:
                    attribute_ruler = self.nlp.add_pipe('attribute_ruler')
                    print("✅ Добавлен attribute_ruler")
                except Exception as e:
                    print(f"⚠️  Не удалось добавить attribute_ruler: {e}")
            
            # Добавляем NER компонент
            if 'ner' not in self.nlp.pipe_names:
                try:
                    ner = self.nlp.add_pipe('ner')
                    print("✅ Добавлен NER")
                except Exception as e:
                    print(f"⚠️  Не удалось добавить NER: {e}")
            
            print("✅ Создана пустая русская модель с базовыми компонентами")
        
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации spaCy: {e}")
        
        # Инициализируем Matcher
        try:
            self.matcher = Matcher(self.nlp.vocab)
            print("✅ Matcher инициализирован")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации Matcher: {e}")
            self.matcher = None
        
        # Добавляем entity ruler только если есть Matcher
        if self.matcher is not None:
            try:
                if 'entity_ruler' not in self.nlp.pipe_names:
                    ruler = self.nlp.add_pipe('entity_ruler', before='ner' if 'ner' in self.nlp.pipe_names else 'last')
                else:
                    ruler = self.nlp.get_pipe('entity_ruler')
                    
                # Добавляем правила в ruler (только базовые, без POS)
                self._add_entity_ruler_patterns(ruler)
                print("✅ EntityRuler добавлен")
                
            except Exception as e:
                print(f"⚠️  Не удалось добавить EntityRuler: {e}")
    
    def _add_entity_ruler_patterns(self, ruler):
        """Добавляет правила в EntityRuler (упрощенные, без POS зависимостей)."""
        patterns = []
        
        # Простые паттерны для возраста (без POS зависимостей)
        age_patterns = [
            {"label": "AGE", "pattern": [
                {"LIKE_NUM": True},
                {"LOWER": "лет"}
            ]},
            {"label": "AGE", "pattern": [
                {"LOWER": "мне"},
                {"LIKE_NUM": True},
                {"LOWER": "лет"}
            ]},
            {"label": "AGE", "pattern": [
                {"LOWER": "возраст"},
                {"TEXT": ":"},
                {"LIKE_NUM": True}
            ]},
        ]
        
        # Простые паттерны для телефонов
        phone_patterns = [
            {"label": "PHONE", "pattern": [
                {"TEXT": "+"},
                {"LIKE_NUM": True}
            ]},
            {"label": "PHONE", "pattern": [
                {"LOWER": "тел"},
                {"TEXT": ":"},
                {"LIKE_NUM": True}
            ]},
        ]
        
        # Простые паттерны для образования (без POS)
        education_patterns = [
            {"label": "EDUCATION", "pattern": [
                {"LOWER": "выпускник"},
                {"IS_ALPHA": True}
            ]},
            {"label": "EDUCATION", "pattern": [
                {"LOWER": "студент"},
                {"IS_ALPHA": True}
            ]},
            {"label": "EDUCATION", "pattern": [
                {"LOWER": "окончил"},
                {"IS_ALPHA": True}
            ]},
        ]
        
        # Объединяем все паттерны
        patterns.extend(age_patterns)
        patterns.extend(phone_patterns)
        patterns.extend(education_patterns)
        
        if patterns:
            ruler.add_patterns(patterns)
            print(f"✅ Добавлено {len(patterns)} упрощенных паттернов в EntityRuler")
    
    def _init_patterns(self):
        """Инициализация паттернов для Matcher (только если Matcher доступен)."""
        if self.matcher is None:
            print("⚠️  Matcher недоступен, пропускаю инициализацию паттернов")
            return
        
        # Простые паттерны без POS зависимостей
        simple_patterns = []
        
        # Паттерны для работы (без POS)
        work_patterns = [
            [{"LOWER": "работаю"}, {"IS_ALPHA": True}],
            [{"LOWER": "должность"}, {"TEXT": ":"}, {"IS_ALPHA": True}],
            [{"LOWER": "профессия"}, {"TEXT": ":"}, {"IS_ALPHA": True}],
        ]
        
        # Паттерны для адресов (без POS)
        address_patterns = [
            [{"LOWER": "живу"}, {"LOWER": {"IN": ["в", "на"]}}, {"IS_ALPHA": True}],
            [{"LOWER": "адрес"}, {"TEXT": ":"}, {"IS_ALPHA": True}],
            [{"LOWER": "улица"}, {"IS_ALPHA": True}],
        ]
        
        try:
            # Добавляем паттерны в matcher
            if work_patterns:
                self.matcher.add("WORK", work_patterns)
                print(f"✅ Добавлено {len(work_patterns)} паттернов WORK")
            
            if address_patterns:
                self.matcher.add("ADDRESS", address_patterns)
                print(f"✅ Добавлено {len(address_patterns)} паттернов ADDRESS")
                
        except Exception as e:
            print(f"⚠️  Ошибка добавления паттернов в Matcher: {e}")
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Извлекает все сущности из текста."""
        if self.nlp is None:
            raise RuntimeError("spaCy модель не инициализирована")
        
        entities = []
        
        try:
            # Обрабатываем текст через spaCy
            doc = self.nlp(text)
            
            # 1. Стандартные NER сущности spaCy
            for ent in doc.ents:
                entity_type = self._map_spacy_type(ent.label_)
                if entity_type:
                    entities.append((
                        ent.start_char,
                        ent.end_char,
                        entity_type,
                        ent.text
                    ))
            
            # 2. Сущности из Matcher (только если доступен)
            if self.matcher:
                try:
                    matches = self.matcher(doc)
                    for match_id, start, end in matches:
                        span = doc[start:end]
                        label = self.nlp.vocab.strings[match_id]
                        entities.append((
                            span.start_char,
                            span.end_char,
                            label,
                            span.text
                        ))
                except Exception as e:
                    print(f"⚠️  Ошибка извлечения из Matcher: {e}")
            
        except Exception as e:
            print(f"⚠️  Ошибка обработки текста в spaCy: {e}")
            return []
        
        # Убираем дубликаты и пересечения
        entities = self._remove_overlapping(entities)
        return sorted(entities, key=lambda x: x[0])
    
    def _map_spacy_type(self, spacy_type: str) -> Optional[str]:
        """Маппинг типов spaCy в наши типы."""
        mapping = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'MONEY': 'MONEY',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'PERCENT': 'PERCENT',
            'PHONE': 'PHONE',
            'EMAIL': 'EMAIL',
            'AGE': 'AGE',
            'EDUCATION': 'EDUCATION',
            'FAMILY': 'FAMILY',
            'WORK': 'POSITION',
            'ADDRESS': 'ADDRESS',
            'MEDICAL': 'MEDICAL',
            'PASSPORT': 'PASSPORT',
            'SNILS': 'SNILS',
            'INN': 'INN',
        }
        
        return mapping.get(spacy_type)
    
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


class SpacyTrainableAnonymizer(SpacyAnonymizer):
    """Анонимизатор spaCy с возможностью дообучения на собственных данных."""
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None,
        model_name: str = "ru_core_news_sm"
    ):
        super().__init__(aggressiveness, anonymization_func, model_name)
        
        # Данные для обучения
        self.training_data = []
    
    def add_training_example(self, text: str, entities: List[Tuple[int, int, str]]):
        """
        Добавляет пример для обучения.
        
        Args:
            text: Текст для обучения
            entities: List[(start, end, label)] - аннотированные сущности
        """
        # Преобразуем в формат spaCy
        spacy_entities = []
        for start, end, label in entities:
            spacy_entities.append((start, end, label))
        
        self.training_data.append((text, {"entities": spacy_entities}))
    
    def train(self, n_iter: int = 30):
        """
        Дообучает модель на добавленных примерах.
        
        Args:
            n_iter: Количество итераций обучения
        """
        if not self.training_data or self.nlp is None:
            print("Нет данных для обучения или модель не инициализирована")
            return
        
        print(f"Начинаю обучение на {len(self.training_data)} примерах...")
        
        # Получаем NER компонент
        ner = self.nlp.get_pipe("ner")
        
        # Добавляем новые метки
        for text, annotations in self.training_data:
            for start, end, label in annotations["entities"]:
                ner.add_label(label)
        
        # Отключаем другие компоненты для ускорения
        with self.nlp.disable_pipes(*[pipe for pipe in self.nlp.pipe_names if pipe != "ner"]):
            # Обучение
            optimizer = self.nlp.resume_training()
            
            for i in range(n_iter):
                losses = {}
                for text, annotations in self.training_data:
                    self.nlp.update([text], [annotations], sgd=optimizer, losses=losses)
                
                if i % 10 == 0:
                    print(f"Итерация {i}, потери: {losses}")
        
        print("Обучение завершено!")
    
    def save_model(self, path: str):
        """Сохраняет обученную модель."""
        if self.nlp:
            self.nlp.to_disk(path)
            print(f"Модель сохранена: {path}")
    
    def load_model(self, path: str):
        """Загружает сохраненную модель."""
        try:
            self.nlp = spacy.load(path)
            print(f"Модель загружена: {path}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")


def create_training_examples() -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """Создает примеры обучающих данных для русских ПД."""
    examples = [
        (
            "Меня зовут Иван Петров, мой телефон +7 916 123 45 67",
            [(11, 22, "PERSON"), (34, 50, "PHONE")]
        ),
        (
            "Мне 25 лет, работаю программистом в Яндексе",
            [(4, 10, "AGE"), (19, 32, "POSITION"), (35, 42, "ORGANIZATION")]
        ),
        (
            "Мой паспорт 45 12 345678, СНИЛС 123-456-789 12",
            [(12, 25, "PASSPORT"), (34, 48, "SNILS")]
        ),
        (
            "Живу в Москве на ул. Ленина, д. 15, кв. 10",
            [(8, 14, "LOCATION"), (17, 43, "ADDRESS")]
        ),
        (
            "Моя жена Анна окончила МГУ в 2020 году",
            [(4, 8, "FAMILY"), (9, 13, "PERSON"), (23, 26, "EDUCATION")]
        ),
    ]
    
    return examples 