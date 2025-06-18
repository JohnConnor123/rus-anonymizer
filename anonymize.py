"""anonymize.py
Анонимизация русского текста с использованием специализированных распознавателей.

Пример использования:
$ echo "Меня зовут Иван Иванов, телефон +7 916 123 45 67" | python anonymize.py
"""

import sys
import re
from typing import Dict, Optional, List

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, EntityRecognizer, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError as e:
    print(f"Ошибка импорта Presidio: {e}")
    print("Установите: pip install presidio-analyzer presidio-anonymizer")
    sys.exit(1)


class RussianPersonRecognizer(EntityRecognizer):
    """Распознаватель русских ФИО через регулярные выражения."""
    
    def __init__(self):
        super().__init__(
            supported_entities=["PERSON"],
            supported_language="en",  # Используем "en" для совместимости
            name="RUSSIAN_PERSON",
            version="0.1"
        )
        # Улучшенные паттерны для русских имен
        self.patterns = [
            # Имя Фамилия (с заглавными буквами)
            r'\b[А-ЯЁ][а-яё]{2,15}\s+[А-ЯЁ][а-яё]{2,20}\b',
            # Имя Отчество Фамилия
            r'\b[А-ЯЁ][а-яё]{2,15}\s+[А-ЯЁ][а-яё]{2,15}\s+[А-ЯЁ][а-яё]{2,20}\b',
            # Фамилия И.О.
            r'\b[А-ЯЁ][а-яё]{2,20}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b',
            # И.О. Фамилия
            r'\b[А-ЯЁ]\.\s*[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]{2,20}\b'
        ]

    def load(self):
        """Обязательный метод для Presidio."""
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None):
        """Анализ текста для поиска русских имен."""
        results = []
        
        for pattern in self.patterns:
            for match in re.finditer(pattern, text):
                # Проверяем, что это не обычные слова
                name_text = match.group().strip()
                if self._is_likely_name(name_text):
                    results.append(
                        RecognizerResult(
                            entity_type="PERSON",
                            start=match.start(),
                            end=match.end(),
                            score=0.85
                        )
                    )
        
        return results
    
    def _is_likely_name(self, text: str) -> bool:
        """Проверяет, похож ли текст на имя."""
        # Исключаем некоторые общие слова
        excluded_words = {
            'Меня зовут', 'Его зовут', 'Её зовут', 'Их зовут',
            'Мой друг', 'Моя подруга', 'Наш коллега', 'Наша коллега'
        }
        
        return text not in excluded_words and len(text.split()) <= 3


class RussianPhoneRecognizer(EntityRecognizer):
    """Распознаватель российских телефонных номеров."""
    
    def __init__(self):
        super().__init__(
            supported_entities=["PHONE_NUMBER"],
            supported_language="en",
            name="RUSSIAN_PHONE",
            version="0.1"
        )
        # Паттерны для российских номеров
        self.patterns = [
            # +7 XXX XXX XX XX
            r'\+7\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}',
            # 8-XXX-XXX-XX-XX
            r'8[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}[-\s]*\d{2}',
            # 8 (XXX) XXX-XX-XX
            r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}',
            # +7(XXX)XXX-XX-XX
            r'\+7\(\d{3}\)\d{3}[-\s]*\d{2}[-\s]*\d{2}'
        ]

    def load(self):
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None):
        """Анализ текста для поиска телефонных номеров."""
        results = []
        
        for pattern in self.patterns:
            for match in re.finditer(pattern, text):
                results.append(
                    RecognizerResult(
                        entity_type="PHONE_NUMBER",
                        start=match.start(),
                        end=match.end(),
                        score=0.90
                    )
                )
        
        return results


class RussianEmailRecognizer(EntityRecognizer):
    """Распознаватель email адресов."""
    
    def __init__(self):
        super().__init__(
            supported_entities=["EMAIL_ADDRESS"],
            supported_language="en",
            name="RUSSIAN_EMAIL",
            version="0.1"
        )
        # Паттерн для email
        self.pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'

    def load(self):
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None):
        """Анализ текста для поиска email адресов."""
        results = []
        
        for match in re.finditer(self.pattern, text):
            results.append(
                RecognizerResult(
                    entity_type="EMAIL_ADDRESS",
                    start=match.start(),
                    end=match.end(),
                    score=0.95
                )
            )
        
        return results


def create_analyzer() -> AnalyzerEngine:
    """Создает анализатор для русского текста."""
    registry = RecognizerRegistry()
    
    # Добавляем только наши кастомные распознаватели
    registry.add_recognizer(RussianPersonRecognizer())
    registry.add_recognizer(RussianPhoneRecognizer())
    registry.add_recognizer(RussianEmailRecognizer())
    
    return AnalyzerEngine(registry=registry)


def anonymize_text(
    text: str,
    entities_to_anonymize: Optional[List[str]] = None,
    anonymization_config: Optional[Dict[str, OperatorConfig]] = None
) -> str:
    """
    Анонимизирует русский текст, заменяя найденные PII.
    
    Args:
        text: Исходный текст
        entities_to_anonymize: Список типов сущностей для анонимизации (None = все)
        anonymization_config: Конфигурация анонимизации
    
    Returns:
        Анонимизированный текст
    """
    analyzer = create_analyzer()
    anonymizer = AnonymizerEngine()

    # Анализируем текст (используем "en" для совместимости)
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=entities_to_anonymize
    )

    # Конфигурация по умолчанию
    if anonymization_config is None:
        anonymization_config = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<СКРЫТО>"})
        }

    # Анонимизируем
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=anonymization_config
    )

    return anonymized_result.text


def main():
    """CLI интерфейс для анонимизации русского текста."""
    if sys.stdin.isatty():
        print("Введите русский текст для анонимизации (Ctrl+D для завершения):")
    
    try:
        text = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(0)
    
    if not text:
        print("Пустой ввод", file=sys.stderr)
        sys.exit(1)

    try:
        # Конфигурация для русского текста
        config = {
            "PERSON": OperatorConfig("replace", {"new_value": "<ИМЯ>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<ТЕЛЕФОН>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<ДАННЫЕ>"})
        }
        
        result = anonymize_text(text, anonymization_config=config)
        print(result)
        
    except Exception as e:
        print(f"Ошибка при анонимизации: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 