"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

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
