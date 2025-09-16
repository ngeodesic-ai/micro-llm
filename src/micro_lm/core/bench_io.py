from __future__ import annotations
from typing import Any, Dict

class ArtifactWriter:
    def collect(self, *, label: str, mapper: Dict[str, Any], verify: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mapper": mapper,
            "verify": verify,
            "schema": {"v": 1, "keys": ["mapper", "verify"]},
        }
