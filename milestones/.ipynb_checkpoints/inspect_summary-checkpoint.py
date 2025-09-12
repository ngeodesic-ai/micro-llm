#!/usr/bin/env python3
import sys, json, pathlib

def _print_run_block(label, blk):
    print(f"  [{label}]")
    if not blk:
        print("    <missing>")
        return
    print(f"    prompt:   {blk.get('prompt')}")
    if "top1_list" in blk:
        print(f"    top1_list: {blk['top1_list']}")
        print(f"    stable_top1: {blk.get('stable_top1')}")
        if blk.get("top1_list"):
            print(f"    top1(first): {blk['top1_list'][0]}")
    else:
        # M1: single run shape
        print(f"    top1:   {blk.get('top1')}")

    outs = blk.get("outputs") or []
    if outs:
        aux0 = outs[0].get("aux", {}) or {}
        if aux0.get("mapper_confidence") is not None:
            mc = aux0["mapper_confidence"]
            try:
                print(f"    conf(first): {mc:.3f}")
            except Exception:
                print(f"    conf(first): {mc}")
        if aux0.get("prior") is not None:
            print(f"    prior(first): {aux0['prior']}")
    print()

def main(path):
    p = pathlib.Path(path)
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
        print(f"- {name}: ok={ok}")
        if reason:
            print(f"  reason: {reason}")

        # Milestone 3 shape: {baseline: {...}, denoised: {...}}
        if "baseline" in sc or "denoised" in sc:
            _print_run_block("baseline", sc.get("baseline"))
            _print_run_block("denoised", sc.get("denoised"))
            continue

        # Milestone 2 shape: output has top1_list/stable_top1
        out = sc.get("output", {}) or {}
        if "top1_list" in out:
            _print_run_block("runs", out)
            continue

        # Milestone 1 shape: single run with top1
        _print_run_block("run", out)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 milestones/inspect_summary.py <summary.json>")
    main(sys.argv[1])
