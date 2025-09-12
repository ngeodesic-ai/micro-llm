#!/usr/bin/env python3
"""
inspect_summary.py — Pretty-print milestone summary JSONs.

Usage:
  python3 milestones/inspect_summary.py .artifacts/defi_milestone1_summary.json

You can point it at any milestone summary JSON produced by scripts
(e.g., defi_milestone2_summary.json, arc_milestone1_summary.json, etc.)
"""

import sys, json, pathlib

def main(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"error: {p} does not exist")

    data = json.loads(p.read_text())

    print(f"Milestone: {data.get('milestone')}")
    print(f"Status:    {data.get('status')}")
    print(f"Rails:     {data.get('rails')}")
    print(f"T:         {data.get('T')}")
    print()

    for sc in data.get("scenarios", []):
        name = sc.get("name")
        ok = sc.get("ok")
        reason = sc.get("reason", "")
        out = sc.get("output", {})
        aux = out.get("aux", {})

        print(f"- {name}: ok={ok}")
        if reason:
            print(f"  reason: {reason}")
        print(f"  prompt: {out.get('prompt')}")
        print(f"  top1:   {out.get('top1')}")

        prior = aux.get("prior")
        if prior:
            print(f"  prior:  {prior}")

        conf = aux.get("mapper_confidence")
        if conf is not None:
            if isinstance(conf, float):
                print(f"  conf:   {conf:.3f}")
            else:
                print(f"  conf:   {conf}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 tools/inspect_summary.py <summary.json>")
    main(sys.argv[1])
