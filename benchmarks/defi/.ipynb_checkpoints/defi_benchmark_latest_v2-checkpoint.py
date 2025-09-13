#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, csv, time, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional



"""
python3 benchmarks/defi/defi_benchmark_latest_v2.py \
  --rails stage11 --runs 20 --seed 42 --fail_on_unstable --csv_with_aux \
  --suite benchmarks/defi/defi_bench_suite_exec8.json \
  --policy '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7},"rails":{"denoise":true,"denoise_mode":"hybrid"}}' \
  --context '{"risk":{"hf":1.3},"oracle":{"age_sec":5,"max_age_sec":30}}' \
  --out_csv .artifacts/defi_bench.csv --out_json .artifacts/defi_bench.json

python3 benchmarks/defi/defi_benchmark_latest_v2.py \
  --rails stage11 --runs 20 --seed 42 --fail_on_unstable --csv_with_aux \
  --suite benchmarks/defi/defi_bench_suite_exec8.json \
  --policy '{"ltv_max":0.75,
           "mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7},
           "rails":{"denoise":false,"denoise_mode":"off"}}' \
  --context '{"risk":{"hf":1.3},"oracle":{"age_sec":5,"max_age_sec":30}}' \
  --out_csv .artifacts/defi_bench.csv --out_json .artifacts/defi_bench.json





--policy '{"...","rails":{"denoise":true,"denoise_mode":"hybrid"}}'
"""
from micro_llm.pipelines.runner import run_micro

DEFAULT_SUITE = [{"name":"deposit_eth","prompt":"deposit 10 ETH into aave","label":"deposit_asset"},
                 {"name":"swap_eth_usdc","prompt":"swap 2 ETH for USDC","label":"swap_asset"},
                 {"name":"withdraw_high_ltv","prompt":"withdraw 5 ETH","label":None},
                 {"name":"borrow_low_hf","prompt":"borrow 1000 USDC","label":None},
                 {"name":"nonexec_abstain","prompt":"check balance","label":None}]

def build_argparser():
    p = argparse.ArgumentParser(description="DeFi Stage-11 benchmark (v2)")
    p.add_argument("--rails", default="stage11")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=180)
    p.add_argument("--policy",  default='{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}')
    p.add_argument("--context", default='{"oracle":{"age_sec":5,"max_age_sec":30}}')
    p.add_argument("--suite", dest="suite_path", default=None)
    p.add_argument("--out_csv", default=".artifacts/defi_bench.csv")
    p.add_argument("--out_json", default=".artifacts/defi_bench.json")
    p.add_argument("--out_report", default=".artifacts/defi_bench_report.md")
    p.add_argument("--denoise_ab", action="store_true")
    p.add_argument("--fail_on_unstable", action="store_true")
    p.add_argument("--csv_with_aux", action="store_true", help="add prior_argmax and mapper_conf to CSV if available")
    return p

def _load_suite(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path: return DEFAULT_SUITE
    p = Path(path)
    return json.loads(p.read_text())

def _run_once(prompt: str, rails: str, T: int, context: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    res = run_micro("defi", prompt, context=context, policy=policy, rails=rails, T=T)
    seq = (res.get("plan") or {}).get("sequence") or []
    top1 = seq[0] if seq else None
    aux = res.get("aux") or {}
    prior = aux.get("prior") or {}
    prior_argmax = max(prior, key=prior.get) if prior else None
    return {
        "prompt": prompt,
        "top1": top1,
        "flags": res.get("flags") or {},
        "verify": res.get("verify") or {},
        "aux": {"prior_argmax": prior_argmax, "mapper_confidence": aux.get("mapper_confidence")}
    }

def _run_stability(prompt, runs, rails, T, context, policy):
    tops=[]; outs=[]
    for _ in range(runs):
        o=_run_once(prompt, rails, T, context, policy); outs.append(o); tops.append(o["top1"])
    stable = len(set(tops))==1
    return {"top1_list": tops, "stable_top1": stable, "outputs": outs}

def _exec_labels(y_true): return sorted({y for y in y_true if y is not None})

def _micro_macro(y_true, y_pred):
    labels=_exec_labels(y_true)
    if not labels:
        return dict(precision_micro=0,recall_micro=0,f1_micro=0,precision_macro=0,recall_macro=0,f1_macro=0)
    per={c:dict(tp=0,fp=0,fn=0) for c in labels}
    for t,p in zip(y_true,y_pred):
        for c in labels:
            if t==c and p==c: per[c]["tp"]+=1
            elif t!=c and p==c: per[c]["fp"]+=1
            elif t==c and p!=c: per[c]["fn"]+=1
    tp=sum(v["tp"] for v in per.values()); fp=sum(v["fp"] for v in per.values()); fn=sum(v["fn"] for v in per.values())
    P=tp/(tp+fp) if tp+fp>0 else 0; R=tp/(tp+fn) if tp+fn>0 else 0; F=(2*P*R)/(P+R) if P+R>0 else 0
    Pm=[]; Rm=[]; Fm=[]
    for c in labels:
        tp_c,fp_c,fn_c=per[c]["tp"],per[c]["fp"],per[c]["fn"]
        p=tp_c/(tp_c+fp_c) if tp_c+fp_c>0 else 0; r=tp_c/(tp_c+fn_c) if tp_c+fn_c>0 else 0; f=(2*p*r)/(p+r) if p+r>0 else 0
        Pm.append(p); Rm.append(r); Fm.append(f)
    return dict(precision_micro=P,recall_micro=R,f1_micro=F,
                precision_macro=sum(Pm)/len(Pm),recall_macro=sum(Rm)/len(Rm),f1_macro=sum(Fm)/len(Fm))

def _rates(y_true, y_pred):
    n=max(1,len(y_true))
    hall=sum(1 for t,p in zip(y_true,y_pred) if t is None and p is not None)/n
    omit=sum(1 for t,p in zip(y_true,y_pred) if t is not None and p is None)/n
    abst=sum(1 for p in y_pred if p is None)/n
    return dict(hallucination_rate=hall,omission_rate=omit,abstain_rate=abst)

def main():
    ap=build_argparser(); ns=ap.parse_args()
    random.seed(ns.seed)
    suite=_load_suite(ns.suite_path)
    policy=json.loads(ns.policy); context=json.loads(ns.context)

    y_true=[]; y_pred=[]; rows=[]; unstable=[]
    for sc in suite:
        st=_run_stability(sc["prompt"], ns.runs, ns.rails, ns.T, context, policy)
        top1_first = st["top1_list"][0] if st["top1_list"] else None
        y_true.append(sc["label"]); y_pred.append(top1_first)
        prior_argmax = (st["outputs"][0]["aux"] or {}).get("prior_argmax")
        mapper_conf  = (st["outputs"][0]["aux"] or {}).get("mapper_confidence")
        row={"name":sc["name"],"prompt":sc["prompt"],"label":sc["label"],
             "top1_first":top1_first,"stable_top1":st["stable_top1"],
             "top1_list":"|".join([x if x is not None else "" for x in st["top1_list"]])}
        if ns.csv_with_aux:
            row.update({"prior_argmax": prior_argmax, "mapper_confidence": mapper_conf})
        rows.append(row)
        if not st["stable_top1"]:
            unstable.append(sc["name"])

    mm=_micro_macro(y_true,y_pred); rates=_rates(y_true,y_pred)
    acc_exact = sum(1 for t,p in zip(y_true,y_pred) if t==p)/max(1,len(y_true))

    # Exec-only metrics
    exec_idx=[i for i,t in enumerate(y_true) if t is not None]
    y_true_exec=[y_true[i] for i in exec_idx]
    y_pred_exec=[y_pred[i] for i in exec_idx]
    mm_exec=_micro_macro(y_true_exec,y_pred_exec) if y_true_exec else {k:0 for k in ["precision_micro","recall_micro","f1_micro","precision_macro","recall_macro","f1_macro"]}
    acc_exec = sum(1 for t,p in zip(y_true_exec,y_pred_exec) if t==p)/max(1,len(y_true_exec)) if y_true_exec else 0

    summary={"samples":len(suite),"runs_per_case":ns.runs,
             "metrics": {"accuracy_exact":acc_exact, **mm, **rates},
             "metrics_exec": {"accuracy_exact":acc_exec, **mm_exec},
             "unstable_cases": unstable}

    # Denoise A/B
    if ns.denoise_ab:
        pol_dn=json.loads(json.dumps(policy)); pol_dn.setdefault("rails", {})["denoise"]=True
        y_pred_dn=[]
        for sc in suite:
            if sc["label"] is None: continue
            st=_run_stability(sc["prompt"], ns.runs, ns.rails, ns.T, context, pol_dn)
            y_pred_dn.append(st["top1_list"][0] if st["top1_list"] else None)
        y_pred_exec = [r["top1_first"] for r in rows if (r["label"] is not None)]
        consistent = len(y_pred_dn)==len(y_pred_exec) and all(a==b for a,b in zip(y_pred_exec,y_pred_dn))
        summary["denoise"]={"exec_top1_consistent": bool(consistent)}

    # Outputs
    Path(".artifacts").mkdir(exist_ok=True, parents=True)
    if ns.out_csv:
        with open(ns.out_csv,"w",newline="") as f:
            cols=list(rows[0].keys())
            w=csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    if ns.out_json:
        with open(ns.out_json,"w") as f: json.dump(summary,f,indent=2)
    if ns.out_report:
        lines=["# DeFi Benchmark Report (v2)","","## Metrics",
               f"- accuracy_exact: {summary['metrics']['accuracy_exact']:.3f}",
               f"- f1_micro: {summary['metrics']['f1_micro']:.3f} • f1_macro: {summary['metrics']['f1_macro']:.3f}",
               f"- hallucination_rate: {summary['metrics']['hallucination_rate']:.3f} • omission_rate: {summary['metrics']['omission_rate']:.3f} • abstain_rate: {summary['metrics']['abstain_rate']:.3f}",
               "","## Exec-only",""
               f"- accuracy_exact: {summary['metrics_exec']['accuracy_exact']:.3f}",
               f"- f1_micro: {summary['metrics_exec']['f1_micro']:.3f} • f1_macro: {summary['metrics_exec']['f1_macro']:.3f}",
               ""]
        Path(ns.out_report).write_text("\n".join(lines))

    # Fail on instability if requested
    ok = True
    if ns.fail_on_unstable and unstable:
        ok = False
    print(json.dumps({"ok": ok, **summary}, indent=2))
    sys.exit(0 if ok else 2)

if __name__=="__main__":
    main()
