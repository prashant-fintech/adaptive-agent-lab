"""Adapter router: decide per query whether the base model answers or the
polite adapter does.

The rules are data, not code - add a row to POLITE_PATTERNS and you have a
new routing behaviour. The decision object carries which rule fired, so a
routing choice is always explainable.
"""

import re
from dataclasses import dataclass

POLITE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("frustration", re.compile(r"\b(frustrat\w*|annoy\w*|angry|hate|stuck|ugh|argh)\b", re.I)),
    ("distress", re.compile(r"\b(doesn'?t work|not working|broken|keeps? failing|give up)\b", re.I)),
    ("gratitude", re.compile(r"\b(thank\w*|appreciate|grateful)\b", re.I)),
    ("courtesy", re.compile(r"\b(please|kindly|would you mind)\b", re.I)),
    ("greeting", re.compile(r"\b(good (morning|afternoon|evening)|hope you|how are you)\b", re.I)),
    ("apology", re.compile(r"\b(sorry|apolog\w*)\b", re.I)),
]


@dataclass
class RouteDecision:
    adapter: str | None  # None -> base model
    rule: str | None     # which pattern fired (None when none did)

    @property
    def label(self) -> str:
        return f"polite adapter (rule: {self.rule})" if self.adapter else "base model"


def route(query: str) -> RouteDecision:
    for rule_name, pattern in POLITE_PATTERNS:
        if pattern.search(query):
            return RouteDecision(adapter="polite", rule=rule_name)
    return RouteDecision(adapter=None, rule=None)
