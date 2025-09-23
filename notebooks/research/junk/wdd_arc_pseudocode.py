# 0) prompt and grid
prompt = "flip the grid horizontally"
grid   = ...  # 12x12 ints

# 1) mapper (text → probs) — but DO NOT touch Stage-11 thresholds with this
p_text = mapper(prompt)  # {"flip_h":0.92, "flip_v":0.05, "rotate":0.03}
y_hat  = max(p_text, key=p_text.get)

# 2) family routing (include a foil)
FAMILY = {
    "flip_h": {"flip_h","flip_v"},
    "flip_v": {"flip_h","flip_v"},
    "rotate": {"rotate","flip_h"},  # rotate vs a flip foil
}
route = FAMILY.get(y_hat, {"flip_h","flip_v","rotate"})

# 3) encode grid → traces (text never enters WDD)
traces_all = traces_from_grid(grid, warp)              # your encoder
traces = {k: traces_all[k] for k in route}             # shortlist only

# 4) Stage-11 with fixed, pre-fit priors & thresholds (no per-example changes)
#    tau_rel, tau_abs_q, null_K remain the same for everyone.
priors = load_priors("stage11_funnel_priors.npz")      # or disable priors if you prefer apples-to-apples
keep, order = geodesic_parse_with_prior(
    traces, priors,
    sigma=9, proto_width=160,
    alpha=0.05, beta_s=0.25, q_s=2,       # prior coupling ON or set use_funnel_prior=0 path
    tau_rel=0.60, tau_abs_q=0.93, null_K=128, seed=42
)  # fixed dual-gate parser; absolute gate uses permutation nulls. :contentReference[oaicite:2]{index=2}

# 5) pre-registered verdict (no easing)
if len(keep) == 1:
    verdict = ("PASS", keep[0])
else:
    verdict = ("ABSTAIN", None)
