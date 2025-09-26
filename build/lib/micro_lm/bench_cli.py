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

import argparse, json, sys, time, os
from typing import List, Dict, Any, Tuple
from collections import Counter
from micro_lm.core.runner import run_micro


def _read_lines(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_bool(x: Any, default=None):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"1", "true", "yes", "y"}:
            return True
        if t in {"0", "false", "no", "n"}:
            return False
    return default


def main(argv=None):
    ap = argparse.ArgumentParser("micro-lm-bench")
    ap.add_argument("domain", choices=["defi", "arc"])
    ap.add_argument("--file", required=True, help="JSONL with rows: {prompt, context?, policy?, backend?, rails?, T?, expect_ok?, expect_label?}")
    ap.add_argument("--out", default=".artifacts/bench_results.jsonl")
    ap.add_argument("--summary-out", default=None, help="Optional path to write the summary JSON")
    ap.add_argument("--default-backend", default="sbert")
    ap.add_argument("--default-rails", default="stage11")
    ap.add_argument("--default-T", type=int, default=180)
    ap.add_argument("--gate-metric", default=None, choices=["ok_acc","label_acc","expect_ok_acc","exact_acc"], help="Optional metric to gate on")
    ap.add_argument("--gate-min", type=float, default=None, help="Minimum value for gate metric, e.g. 0.66")
    args = ap.parse_args(argv)

    rows = _read_lines(args.file)

    # tallies
    total = 0
    ok_cnt = 0
    label_hits = 0
    label_den = 0
    expect_ok_hits = 0
    expect_ok_den = 0
    exact_hits = 0
    exact_den = 0
    conf = Counter()  # (pred_label, expect_label) -> count

    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as w:
        for r in rows:
            total += 1
            out = run_micro(
                args.domain,
                r["prompt"],
                context=r.get("context", {}),
                policy=r.get("policy", {}),
                rails=r.get("rails", args.default_rails),
                T=int(r.get("T", args.default_T)),
                backend=r.get("backend", args.default_backend),
            )

            # base ok
            ok = bool(out.get("ok"))
            ok_cnt += 1 if ok else 0

            # optional expectations
            exp_ok = _safe_bool(r.get("expect_ok"), default=None)
            exp_label = r.get("expect_label")

            # label accuracy (only if expect_label provided)
            pred_label = out.get("label")
            if exp_label is not None:
                label_den += 1
                if pred_label == exp_label:
                    label_hits += 1
                conf[(str(pred_label), str(exp_label))] += 1

            # expected-ok accuracy (only if expect_ok provided)
            if exp_ok is not None:
                expect_ok_den += 1
                if ok is exp_ok:
                    expect_ok_hits += 1

            # exact match (need both)
            if (exp_ok is not None) and (exp_label is not None):
                exact_den += 1
                if (ok is exp_ok) and (pred_label == exp_label):
                    exact_hits += 1

            # Write per-row result with evaluation fields (where applicable)
            eval_fields = {
                "ok": ok,
                "label": pred_label,
            }
            if exp_ok is not None:
                eval_fields["expect_ok"] = exp_ok
                eval_fields["ok_match"] = (ok is exp_ok)
            if exp_label is not None:
                eval_fields["expect_label"] = exp_label
                eval_fields["label_match"] = (pred_label == exp_label)

            w.write(json.dumps({"input": r, "output": out, "eval": eval_fields}) + "\n")

    dt = time.time() - t0

    # compute metrics
    ok_acc = ok_cnt / total if total else 0.0
    label_acc = label_hits / label_den if label_den else None
    expect_ok_acc = expect_ok_hits / expect_ok_den if expect_ok_den else None
    exact_acc = exact_hits / exact_den if exact_den else None

    # tidy confusion
    confusion = [{"pred": p, "expect": e, "count": c} for (p, e), c in sorted(conf.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))]

    summary = {
        "total": total,
        "ok": ok_cnt,
        "ok_acc": ok_acc,
        "label_acc": label_acc,
        "expect_ok_acc": expect_ok_acc,
        "exact_acc": exact_acc,
        "sec": dt,
        "confusion": confusion,
        "denominators": {"label": label_den, "expect_ok": expect_ok_den, "exact": exact_den},
    }

    # print summary and optionally write it
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
        with open(args.summary_out, "w") as f:
            json.dump(summary, f)

    # optional gate
    if args.gate_metric and args.gate_min is not None:
        val = summary.get(args.gate_metric)
        # if metric is None (no denominator), treat as fail
        ok_gate = (val is not None) and (float(val) >= float(args.gate_min))
        if not ok_gate:
            print(json.dumps({"gate_metric": args.gate_metric, "value": val, "min": args.gate_min, "result": "FAIL"}))
            sys.exit(1)
        else:
            print(json.dumps({"gate_metric": args.gate_metric, "value": val, "min": args.gate_min, "result": "PASS"}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
