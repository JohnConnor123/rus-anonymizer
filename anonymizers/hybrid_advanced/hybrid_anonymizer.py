"""
Гибридный анонимизатор, объединяющий несколько подходов.
Использует только доступные библиотеки без DeepPavlov.
"""

from typing import List, Tuple, Optional, Callable, Dict, Any
import statistics

# Импортируем базовый класс
from ..base import BaseAnonymizer

# Импортируем доступные анонимизаторы (без DeepPavlov)
from ..natasha_enhanced.enhanced_natasha import EnhancedNatashaAnonymizer
from ..spacy_extended.spacy_anonymizer import SpacyAnonymizer  
from ..transformers_bert.bert_anonymizer import BertAnonymizer
from ..regexp_baseline.regexp_anonymizer import RegExpBaselineAnonymizer

# Убираем импорт DeepPavlov
# from ..deeppavlov_ner.deeppavlov_anonymizer import DeepPavlovAnonymizer, DeepPavlovMultiModelAnonymizer


class HybridAdvancedAnonymizer(BaseAnonymizer):
    """
    Гибридный анонимизатор, объединяющий несколько подходов.
    Без использования DeepPavlov для совместимости.
    """
    
    def __init__(
        self,
        aggressiveness: float = 0.8,
        anonymization_func: Optional[Callable[[str, str], str]] = None,
        use_consensus: bool = True,
        consensus_threshold: float = 0.4
    ):
        super().__init__(aggressiveness, anonymization_func)
        
        self.use_consensus = use_consensus
        self.consensus_threshold = consensus_threshold
        
        # Словарь анонимизаторов и их весов
        self.anonymizers = {}
        self.anonymizer_weights = {}
        
        # Инициализируем доступные анонимизаторы
        self._init_anonymizers()
    
    def _init_anonymizers(self):
        """Инициализация всех доступных анонимизаторов (без DeepPavlov)."""
        print("🔧 Инициализация гибридного анонимизатора...")
        
        # 1. Natasha Enhanced (обычно работает стабильно)
        try:
            self.anonymizers['natasha'] = EnhancedNatashaAnonymizer(
                aggressiveness=self.aggressiveness
            )
            self.anonymizer_weights['natasha'] = 1.0
            print("✅ Natasha Enhanced загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки Natasha: {e}")
        
        # 2. spaCy Extended (может иметь проблемы с моделью)
        try:
            self.anonymizers['spacy'] = SpacyAnonymizer(
                aggressiveness=self.aggressiveness
            )
            self.anonymizer_weights['spacy'] = 1.1
            print("✅ SpaCy Extended загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки SpaCy: {e}")
        
        # 3. BERT Transformers (может требовать много памяти)
        try:
            self.anonymizers['bert'] = BertAnonymizer(
                aggressiveness=self.aggressiveness
            )
            self.anonymizer_weights['bert'] = 1.3
            print("✅ BERT Transformers загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки BERT: {e}")
        
        # 4. RegExp Baseline (всегда должен работать)
        try:
            self.anonymizers['regexp'] = RegExpBaselineAnonymizer(
                aggressiveness=self.aggressiveness
            )
            self.anonymizer_weights['regexp'] = 0.7
            print("✅ RegExp Baseline загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки RegExp: {e}")
        
        print(f"🎯 Гибридный анонимизатор готов с {len(self.anonymizers)} компонентами")
        
        if not self.anonymizers:
            raise RuntimeError("❌ Не удалось загрузить ни одного анонимизатора!")
    
    def extract_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Извлекает сущности используя все доступные анонимизаторы.
        Применяет консенсус или простое объединение.
        """
        if not self.anonymizers:
            raise RuntimeError("Анонимизаторы не инициализированы")
        
        all_results = {}
        
        # Получаем результаты от всех анонимизаторов
        for name, anonymizer in self.anonymizers.items():
            try:
                entities = anonymizer.extract_entities(text)
                all_results[name] = entities
                print(f"📊 {name}: найдено {len(entities)} сущностей")
            except Exception as e:
                print(f"❌ Ошибка в {name}: {e}")
                all_results[name] = []
        
        # Объединяем результаты
        if self.use_consensus:
            return self._consensus_merge(all_results, text)
        else:
            return self._simple_merge(all_results)
    
    def _consensus_merge(self, all_results: Dict[str, List], text: str) -> List[Tuple[int, int, str, str]]:
        """
        Объединяет результаты на основе консенсуса.
        Сущность принимается только если её нашло достаточно анонимизаторов.
        """
        entity_votes = {}  # (start, end, type) -> список источников
        
        # Собираем голоса
        for anonymizer_name, entities in all_results.items():
            weight = self.anonymizer_weights.get(anonymizer_name, 1.0)
            
            for start, end, entity_type, value in entities:
                key = (start, end, entity_type)
                
                if key not in entity_votes:
                    entity_votes[key] = []
                
                entity_votes[key].append({
                    'source': anonymizer_name,
                    'weight': weight,
                    'value': value
                })
        
        # Фильтруем по консенсусу
        final_entities = []
        total_weight = sum(self.anonymizer_weights.values())
        
        for (start, end, entity_type), votes in entity_votes.items():
            # Вычисляем суммарный вес голосов
            vote_weight = sum(vote['weight'] for vote in votes)
            consensus_score = vote_weight / total_weight
            
            if consensus_score >= self.consensus_threshold:
                # Выбираем лучшее значение (от самого надёжного источника)
                best_vote = max(votes, key=lambda x: x['weight'])
                
                final_entities.append((
                    start, end, entity_type, best_vote['value']
                ))
        
        # Убираем пересечения
        final_entities = self._remove_overlapping(final_entities)
        
        print(f"🤝 Консенсус: {len(final_entities)} сущностей прошли порог {self.consensus_threshold}")
        return sorted(final_entities, key=lambda x: x[0])
    
    def _simple_merge(self, all_results: Dict[str, List]) -> List[Tuple[int, int, str, str]]:
        """Простое объединение всех результатов с удалением дубликатов."""
        all_entities = []
        
        # Объединяем все результаты
        for anonymizer_name, entities in all_results.items():
            for entity in entities:
                all_entities.append(entity)
        
        # Убираем дубликаты и пересечения
        unique_entities = self._remove_overlapping(all_entities)
        
        print(f"🔗 Простое объединение: {len(unique_entities)} уникальных сущностей")
        return sorted(unique_entities, key=lambda x: x[0])
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по используемым анонимизаторам."""
        stats = {
            'total_anonymizers': len(self.anonymizers),
            'available_anonymizers': list(self.anonymizers.keys()),
            'weights': self.anonymizer_weights.copy(),
            'consensus_enabled': self.use_consensus,
            'consensus_threshold': self.consensus_threshold if self.use_consensus else None
        }
        
        return stats


class HybridAdvancedAnalyzer:
    """Анализатор для исследования работы гибридного анонимизатора."""
    
    def __init__(self):
        self.anonymizers = {}
        self._init_all_anonymizers()
    
    def _init_all_anonymizers(self):
        """Инициализация всех доступных анонимизаторов для анализа."""
        print("🔬 Инициализация анонимизаторов для анализа...")
        
        # 1. Natasha Enhanced
        try:
            self.anonymizers['natasha'] = EnhancedNatashaAnonymizer(aggressiveness=0.8)
            print("✅ Natasha Enhanced")
        except Exception as e:
            print(f"❌ Natasha Enhanced: {e}")
        
        # 2. spaCy Extended
        try:
            self.anonymizers['spacy'] = SpacyAnonymizer(aggressiveness=0.8)
            print("✅ SpaCy Extended")
        except Exception as e:
            print(f"❌ SpaCy Extended: {e}")
        
        # 3. BERT Transformers
        try:
            self.anonymizers['bert'] = BertAnonymizer(aggressiveness=0.8)
            print("✅ BERT Transformers")
        except Exception as e:
            print(f"❌ BERT Transformers: {e}")
        
        # 4. RegExp Baseline
        try:
            self.anonymizers['regexp'] = RegExpBaselineAnonymizer(aggressiveness=0.8)
            print("✅ RegExp Baseline")
        except Exception as e:
            print(f"❌ RegExp Baseline: {e}")
        
        print(f"📊 Готово для анализа: {len(self.anonymizers)} анонимизаторов")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Подробный анализ извлечения сущностей из текста."""
        results = {}
        
        for name, anonymizer in self.anonymizers.items():
            try:
                entities = anonymizer.extract_entities(text)
                results[name] = {
                    'entities': entities,
                    'count': len(entities),
                    'types': list(set(e[2] for e in entities))
                }
            except Exception as e:
                results[name] = {
                    'error': str(e),
                    'entities': [],
                    'count': 0,
                    'types': []
                }
        
        return results
    
    def compare_performance(self, test_texts: List[str]) -> Dict[str, Any]:
        """Сравнивает производительность анонимизаторов на множестве текстов."""
        performance = {}
        
        for name in self.anonymizers.keys():
            performance[name] = {
                'total_entities': 0,
                'avg_entities_per_text': 0,
                'entity_types': set(),
                'success_rate': 0,
                'errors': 0
            }
        
        for text in test_texts:
            for name, anonymizer in self.anonymizers.items():
                try:
                    entities = anonymizer.extract_entities(text)
                    performance[name]['total_entities'] += len(entities)
                    performance[name]['entity_types'].update(e[2] for e in entities)
                except Exception as e:
                    performance[name]['errors'] += 1
        
        # Вычисляем средние значения
        total_texts = len(test_texts)
        for name in performance.keys():
            perf = performance[name]
            perf['avg_entities_per_text'] = perf['total_entities'] / total_texts
            perf['success_rate'] = (total_texts - perf['errors']) / total_texts
            perf['entity_types'] = list(perf['entity_types'])
        
        return performance 