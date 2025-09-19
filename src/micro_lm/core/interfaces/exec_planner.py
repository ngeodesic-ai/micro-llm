from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

@dataclass
class PlanStep:
    op: str
    args: Dict[str, Any]

@dataclass
class Plan:
    steps: List[PlanStep]
    rationale: str

class ExecPlanner(Protocol):
    def plan(self, *, intent: str, text: str, context: Dict[str, Any]) -> Plan:
        """Translate intent => concrete, explainable plan."""
        ...
