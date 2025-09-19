#!/usr/bin/env python3
import sys, json, glob, os, math

def load(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    paths = sys.argv[1:] or glob.glob(".artifacts/defi/audit_bench*/metrics_audit.json")
    if not paths:
        print("No metrics_audit.json found", file=sys.stderr)
        sys.exit(1)
    print("suite,coverage,abstain,hallu,multi,span_yield,tau_rel,tau_abs")
    for p in sorted(paths):
        m = load(p)
        params = m.get("params", {})
        print(",".join([
            os.path.dirname(p),
            f"{m.get('coverage',0):.4f}",
            f"{m.get('abstain_rate',0):.4f}",
            f"{m.get('hallucination_rate',0):.4f}",
            f"{m.get('multi_accept_rate',0):.4f}",
            f"{m.get('span_yield_rate',0):.4f}",
            f"{params.get('tau_rel',0):.2f}",
            f"{params.get('tau_abs',0):.2f}",
        ]))
if __name__ == "__main__":
    main()
