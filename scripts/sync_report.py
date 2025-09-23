#!/usr/bin/env python3
import json, pathlib, sys

"""
# Save nb_vs_pkg output:
python3 scripts/nb_vs_pkg.py > .artifacts/pkg_map/nb_vs_pkg.json
# Build combined report:
python3 scripts/sync_report.py
# Open:
open .artifacts/pkg_map/sync_report.md   # or code .artifacts/pkg_map/sync_report.md
"""

ART = pathlib.Path(".artifacts/pkg_map")
PKG = json.loads((ART/"pkg_modules.json").read_text())
EX  = json.loads((ART/"facade_exports.json").read_text())
HIL = json.loads((ART/"highlights.json").read_text())

# nb_vs_pkg.py prints to stdout; allow piping to a file, or re-run it here if needed.
NB_DIFF_PATH = ART / "nb_vs_pkg.json"
if NB_DIFF_PATH.exists():
    NB = json.loads(NB_DIFF_PATH.read_text())
else:
    print("ERROR: .artifacts/pkg_map/nb_vs_pkg.json not found. Pipe nb_vs_pkg.py output there.", file=sys.stderr)
    sys.exit(2)

only_nb = set(NB.get("only_in_notebooks", []))
only_pkg = set(NB.get("only_in_package", []))
inter = set(NB.get("intersection", []))

# 1) Import errors (modules that failed to import)
errors = [m for m in PKG if "error" in m]

# 2) Public API that is NOT re-exported by the facade
pkg_public = set()
for m in PKG:
    for n in m.get("public", []) or []:
        pkg_public.add(n)
facade_gaps = sorted(pkg_public - set(EX))

# 3) Notebook-only symbols that look like registry/runner logic (promote candidates)
promote = sorted([n for n in only_nb if any(k in n.lower() for k in (
    "build", "score", "template", "registry", "family", "runtime", "order", "keep"
))])

# 4) Potential dead code (package symbols used nowhere in notebooks AND not exported)
deadish = sorted([n for n in only_pkg if n not in EX])

# 5) Build a Markdown report
out = []
out.append("# micro_lm ⇄ notebooks sync report\n")
out.append("## Summary\n")
out.append(f"- Notebook-only symbols (candidates to promote): **{len(only_nb)}**")
out.append(f"- Package-only symbols: **{len(only_pkg)}**")
out.append(f"- Facade export count (__all__): **{len(EX)}**")
out.append(f"- Import errors: **{len(errors)}**\n")

out.append("## 1) Promote from notebooks → package\n")
out.append("These appear only in notebooks (likely core WDD/ARC/DeFi logic). Move them into `domains/<domain>/wdd_runtime.py` and wire via `core.audit.run_families`.\n")
out += [f"- [ ] `{n}`" for n in promote or only_nb or []] or ["- (none)"]

out.append("\n## 2) Facade gaps (public but not re-exported)\n")
out.append("For notebook stability, re-export key symbols in `micro_lm/__init__.py` so notebooks import the facade only.")
out += [f"- [ ] `{n}`" for n in facade_gaps] or ["- (none)"]

out.append("\n## 3) Potential dead code (notebooks don’t call these & not in facade)\n")
out += [f"- [ ] `{n}`" for n in deadish] or ["- (none)"]

out.append("\n## 4) Modules with import errors\n")
for m in errors:
    out.append(f"- [ ] {m['module']} — `{m.get('error')}`")

out.append("\n## 5) Highlights (Tier-2 critical paths)\n")
for key, v in HIL.items():
    out.append(f"### {key}\n```json\n{json.dumps(v, indent=2)}\n```")

ART.mkdir(parents=True, exist_ok=True)
(ART/"sync_report.md").write_text("\n".join(out))
print(f"Wrote: {ART/'sync_report.md'}")

# Exit nonzero if there’s anything actionable
exit_code = 0
if promote or facade_gaps or errors:
    exit_code = 1
sys.exit(exit_code)
