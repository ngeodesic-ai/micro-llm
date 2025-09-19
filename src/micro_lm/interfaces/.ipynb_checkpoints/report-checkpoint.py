# interfaces/report.py
from typing import Protocol, Dict, Any, List
from pathlib import Path
import json

class Sink(Protocol):
    def scenario(self, name: str, output: Dict[str, Any], passed: bool) -> None: ...
    def metrics(self, label: str, values: Dict[str, float]) -> None: ...
    def failures(self, items: List[str]) -> None: ...
    def flush(self) -> None: ...

class JsonSink:
    def __init__(self, path: str):
        self.path = Path(path); self.data = {"scenarios": [], "metrics": {}, "failures": []}
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def scenario(self, name, output, passed): self.data["scenarios"].append({"name": name, "passed": passed, "output": output})
    def metrics(self, label, values): self.data["metrics"][label] = values
    def failures(self, items): self.data["failures"].extend(items)
    def flush(self): self.path.write_text(json.dumps(self.data, indent=2))

class MarkdownSink:
    def __init__(self, path: str):
        self.path = Path(path); self.lines: List[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def scenario(self, name, output, passed):
        v = (output.get("verify") or {})
        self.lines.append(f"- **{name}** → `passed={passed}` • top1=`{output.get('top1')}` • verify.ok=`{bool(v.get('ok'))}` • reason=`{v.get('reason','')}`")
    def metrics(self, label, values):
        self.lines.append(f"## Metrics — {label}")
        for k, v in values.items(): self.lines.append(f"- {k}: {v}")
        self.lines.append("")
    def failures(self, items):
        if items:
            self.lines.append("## Failures");  [self.lines.append(f"- {f}") for f in items]; self.lines.append("")
    def flush(self): self.path.write_text("\n".join(self.lines) or "# Report\n")
