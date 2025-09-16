import json, os
from typing import Any

def _load_json_or_yaml(path: str) -> Any:
    if not os.path.exists(path):
        return {}
    if path.endswith((".yml", ".yaml")):
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r") as f:
        return json.load(f)

def discover_context(default="{}"):
    # prefer explicit env, then file, then CLI default
    fn = os.getenv("MICROLM_CONTEXT", "configs/context.yml")
    return _load_json_or_yaml(fn) or json.loads(default)

def discover_policy(default="{}"):
    fn = os.getenv("MICROLM_POLICY", "configs/policy.yml")
    return _load_json_or_yaml(fn) or json.loads(default)
