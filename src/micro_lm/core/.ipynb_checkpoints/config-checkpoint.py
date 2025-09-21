# src/micro_lm/core/config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

def load_domain_config(domain: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configs/{domain}.yaml if present, else returns {}.
    """
    base = Path(base_dir or Path(__file__).resolve().parents[3])  # repo root
    cfg_path = base / "configs" / f"{domain}.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    # normalize keys we care about
    out = {}
    out["audit"] = data.get("audit", {})
    out["T"]     = int(data.get("T", 600))
    out["seed"]  = int(data.get("seed", 0))
    out["defaults"] = data.get("defaults", {})
    out["family_overrides"] = data.get("family_overrides", {})
    out["encoder"] = data.get("encoder", {})
    out["prototypes"] = data.get("prototypes", {})
    out["anchors"] = data.get("anchors", {})
    return out
