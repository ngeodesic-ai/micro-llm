
# Auditor Bench (Single-File Harness)

This harness runs the **auditor-mode** trial that fixes the rails tautology:
- **Text → independent latents** (no mapper coupling).
- **Matched filter + parser** validate evidence.
- Mapper is optional and only **filters channels**, never injects energy.

## Files
- `audit_bench.py` — the harness.
- You provide 3 CSVs: `train.csv`, `test.csv`, `junk.csv`.

### CSV formats
- `train.csv` — `id,prompt,primitive`
- `test.csv`  — `id,prompt,primitive`
- `junk.csv`  — `id,prompt`

## Usage
```bash
python3 audit_bench.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --junk_csv junk.csv \
  --out_dir results_auditor_only \
  --mode auditor_only \
  --tau_map 0.7 --tau_span 0.55 --tau_ood 0.45 \
  --alpha 0.7 --beta 0.3 --T 720 --top_k_phrases 20
```

Optionally compare with mapper filter:
```bash
python3 audit_bench.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --junk_csv junk.csv \
  --out_dir results_mapper_and_auditor \
  --mode mapper_and_auditor
```

## Replace the embedding & mapper
- Replace `PlaceholderEmbedding` with your encoder (e.g., mpnet, BGE, etc.).
- Replace `MapperHook.predict` with your mapper. **Do not** inject energy based on labels.
- The auditor encoder builds **latents from the prompt**:
  - If OOD (`max cos < tau_ood`) → noise-only latents → rails abstain.
  - Else builds modest lobes per primitive proportional to similarity and phrase span scores.

## Outputs
- `metrics.json` — precision/recall/F1 on test, hallucination & abstain rates on junk.
- `preds.csv` — row-by-row predictions.
- `phrases.json` — harvested phrases per primitive.
- `report.md` — brief errors and junk failures.

## Acceptance gates
- OOD hallucinations = **0** (auditor abstains).
- In-domain macro-F1 ≥ **0.95** (target ≥ 0.98 with a good encoder).
- Show at least one case where mapper alone mislabels but auditor abstains.
