import re
import pytest

PD_REGEXPS = [
    re.compile(r"(?:\+7|8)?\s?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}"),
    re.compile(r"[a-zA-Z0-9._%+-]+@[\w.-]+\.[a-zA-Z]{2,}"),
    re.compile(r"\b\d{4}\s?\d{6}\b"),
    re.compile(r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b"),
]

def has_potential_pd(text: str) -> bool:
    """Локальная функция, повторяющая логику _has_potential_pd"""
    return any(r.search(text) for r in PD_REGEXPS)

@pytest.mark.parametrize("text,expected", [
    ("Мой номер телефона +7 916 123 45 67", True),
    ("Напишите мне на ivanov@example.com", True),
    ("Серия паспорта 4004 123456", True),
    ("Номер карты 4276 4000 1234 5678", True),
    ("Сегодня хорошая погода и ничего личного", False),
])
def test_has_potential_pd(text, expected):
    assert has_potential_pd(text) is expected 