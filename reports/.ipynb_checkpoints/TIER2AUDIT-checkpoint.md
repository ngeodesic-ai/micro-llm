# Audit Comparison — Tier-1 vs Tier-2 (micro-lm)

This document compares the **Audit** step in Tier-1 vs Tier-2.  
The **Mapper** is unchanged between tiers; only the **Audit** evolves.

---

## Table

| Aspect | **Tier-1 Audit** | **Tier-2 Audit (WDD / NGF rails)** |
|---|---|---|
| **Purpose** | Fast, simple safety gate | Deterministic, statistically-calibrated detection |
| **Inputs** | `label̂`, `confidence` from mapper + DeFi state (HF/LTV/oracle) | SBERT latents → PCA(d) → per-class prototypes + DeFi state |
| **Signals computed** | Softmax/posterior `p(label̂)` | Similarity margins, residual energies, matched-filter scores, null stats |
| **Decision rule** | `p(label̂) ≥ τ_conf` AND policy checks pass | Dual gates (margin + matched filter vs nulls) AND policy checks pass |
| **Policy checks** | LTV, HF, oracle freshness (rule-based) | Same LTV/HF/oracle rules (unchanged) |
| **Abstain/Block** | Low confidence → abstain; policy fail → block/abstain | Margin below gates or phantom risk → abstain; policy fail → block/abstain |
| **Determinism** | Good, but can flip near τ_conf | Strong: Warp → Detect → Denoise suppresses phantoms |
| **Calibration** | Fixed probability threshold | Null distributions, CFAR/FDR-style gates |
| **Outputs** | `{ok, reason, tags}` | `{ok, reason, tags}` + detection stats |
| **Complexity** | Lightweight | Heavier (PCA, prototypes, filters, denoising) |
| **When to use** | MVP, CI, low-risk | High-stakes flows needing determinism |

---

## Visual Flow

**Tier-1**
```
prompt → SBERT → mapper ──►  [ Audit ]
                              • p(label̂) ≥ τ_conf ?
                              • LTV/HF/oracle ok ?
                           → plan/abstain/block
```

**Tier-2**
```
prompt → SBERT → mapper ──►  [ Audit ]
                              • PCA(d), prototypes
                              • margin & matched filter
                              • null-calibrated gates
                              • LTV/HF/oracle ok ?
                           → plan/abstain/block
```

---

## Pseudocode

**Tier-1**
```python
p, label = mapper.score(prompt)
if p < policy.conf_threshold:
    return abstain("low_conf")
if not policy_checks_ok(state):
    return block("policy_fail")
return accept(label)
```

**Tier-2**
```python
z = sbert.encode(prompt)
y = pca.transform(standardize(z))
margin, mf = detect(y, prototypes)
if not pass_dual_gates(margin, mf, null_stats):
    return abstain("below_gates")
if not policy_checks_ok(state):
    return block("policy_fail")
return accept(label)
```
