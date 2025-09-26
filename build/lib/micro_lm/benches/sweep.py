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
import argparse, json, time
from pathlib import Path
from typing import Dict, Any, Iterable, List
from micro_lm.core.runner import run_micro

def _ts() -> str: return time.strftime("%Y%m%d-%H%M%S")

def _load_cases(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return json.loads(path.read_text())

def run_sweep(*, domain: str, cases: Iterable[Dict[str,Any]], cfg: Dict[str,Any], out_dir: Path) -> Dict[str,Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str,Any]] = []
    for i, case in enumerate(cases):
        out = run_micro(
            domain,
            case.get("prompt",""),
            context=case.get("context",{}),
            policy=cfg.get("policy",{}),
            rails=cfg.get("rails","stage11"),
            T=cfg.get("T", 180),
            backend=cfg.get("mapper","wordmap"),
        )
        rows.append({"i": i, "case": case, "out": out})
    # artifacts
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    hits = sum(1 for r in rows if isinstance(r.get("out"), dict))
    topline = {"n": len(rows), "hits": hits, "hit_rate": (hits/len(rows) if rows else 0.0)}
    (out_dir / "topline.json").write_text(json.dumps(topline, indent=2))
    return topline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["arc","defi"])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mapper", default="wordmap", choices=["wordmap","sbert"])
    ap.add_argument("--audit_backend", default="threshold", choices=["threshold","wdd"])
    ap.add_argument("--mode", default="pure", choices=["pure","family","guided"])
    ap.add_argument("--T", type=int, default=180)
    args = ap.parse_args()

    cases = _load_cases(Path(args.cases))
    policy = {"audit": {"backend": args.audit_backend, "mode": args.mode}}
    cfg = {"policy": policy, "mapper": args.mapper, "T": args.T, "rails": "stage11"}
    out_dir = Path(args.out_dir) / _ts()
    topline = run_sweep(domain=args.domain, cases=cases, cfg=cfg, out_dir=out_dir)
    print(json.dumps(topline))
    raise SystemExit(0)
if __name__ == "__main__":
    main()
