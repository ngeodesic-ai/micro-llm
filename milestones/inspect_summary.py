#!/usr/bin/env python3
"""
inspect_summary.py — Pretty-print milestone summary JSONs.

Usage:
  python3 milestones/inspect_summary.py .artifacts/defi_milestone2_summary.json
  (Works for any milestone summary produced by your scripts.)
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
    if "T" in data: print(f"T:         {data.get('T')}")
    if "runs" in data: print(f"runs:      {data.get('runs')}")
    print()

    for sc in data.get("scenarios", []):
        name = sc.get("name")
        ok = sc.get("ok")
        reason = sc.get("reason", "")
        out = sc.get("output", {}) or {}
        aux = out.get("aux", {}) or {}

        print(f"- {name}: ok={ok}")
        if reason:
            print(f"  reason: {reason}")

        prompt = out.get("prompt")
        if prompt:
            print(f"  prompt: {prompt}")

        # Support both M1 (single top1) and M2 (top1_list across runs)
        top1 = out.get("top1")
        top1_list = out.get("top1_list")
        stable = out.get("stable_top1")

        if top1_list is not None:
            print(f"  top1_list: {top1_list}")
            if stable is not None:
                print(f"  stable_top1: {stable}")
            if top1_list:
                print(f"  top1(first): {top1_list[0]}")
        else:
            print(f"  top1:   {top1}")

        # Try to show aux from first run (if present)
        outs = out.get("outputs") or []
        if outs:
            aux0 = outs[0].get("aux", {}) or {}
            if aux0.get("prior") is not None:
                print(f"  prior(first): {aux0['prior']}")
            if aux0.get("mapper_confidence") is not None:
                mc = aux0["mapper_confidence"]
                print(f"  conf(first):  {mc:.3f}" if isinstance(mc, float) else f"  conf(first):  {mc}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 milestones/inspect_summary.py <summary.json>")
    main(sys.argv[1])
