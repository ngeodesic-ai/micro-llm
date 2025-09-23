#!/usr/bin/env python3
import importlib, inspect, json, pkgutil, sys, pathlib

PKG = "micro_lm"
root = pathlib.Path.cwd()
out_dir = root / ".artifacts" / "pkg_map"
out_dir.mkdir(parents=True, exist_ok=True)

def safe_import(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        return {"__import_error__": repr(e)}

def public_api(obj):
    names = []
    for n, v in sorted(getattr(obj, "__dict__", {}).items()):
        if n.startswith("_"): 
            continue
        if inspect.isfunction(v) or inspect.isclass(v):
            names.append(n)
    return names

def module_path(mod):
    try:
        p = pathlib.Path(inspect.getsourcefile(mod) or inspect.getfile(mod))
        return str(p.resolve())
    except Exception:
        return None

# 1) Map all micro_lm.* modules
modules = []
for finder, name, ispkg in pkgutil.walk_packages([str(root / "src")], prefix=""):
    if not name.startswith(PKG):
        continue
    fq = name
    mod = safe_import(fq)
    entry = {"module": fq, "is_pkg": ispkg}
    if isinstance(mod, dict) and "__import_error__" in mod:
        entry["error"] = mod["__import_error__"]
    else:
        entry["file"] = module_path(mod)
        entry["public"] = public_api(mod)
    modules.append(entry)

# 2) What does micro_lm.__init__ re-export?
facade = safe_import(f"{PKG}")
facade_exports = []
if not isinstance(facade, dict):
    for n in getattr(facade, "__all__", []):
        facade_exports.append(n)

# 3) Quick highlights we care about during Tier-2
highlights = {}
def add_h(path, label):
    try:
        m = importlib.import_module(path)
        highlights[label] = {
            "module": path,
            "file": module_path(m),
            "public": public_api(m),
        }
    except Exception as e:
        highlights[label] = {"module": path, "error": repr(e)}

add_h("micro_lm.domains.defi.families_wdd", "defi_families_wdd")      # present already :contentReference[oaicite:3]{index=3}
add_h("micro_lm.compat.legacy_imports", "legacy_imports_shim")        # present already :contentReference[oaicite:4]{index=4}
# add_h("micro_lm.domains.arc.families_wdd", "arc_families_wdd")      # enable once present

# 4) Write outputs
(out_dir / "pkg_modules.json").write_text(json.dumps(modules, indent=2))
(out_dir / "facade_exports.json").write_text(json.dumps(facade_exports, indent=2))
(out_dir / "highlights.json").write_text(json.dumps(highlights, indent=2))

# 5) Render a quick README.md for human scan
lines = ["# micro_lm package map\n"]
lines.append("## Top-level facade exports (__all__)")
lines.append("```\n" + "\n".join(facade_exports or ["<none>"]) + "\n```")
lines.append("\n## Highlights (Tier-2 relevant)")
for k, v in highlights.items():
    lines.append(f"### {k}\n")
    lines.append("```json\n" + json.dumps(v, indent=2) + "\n```")
lines.append("\n## All modules")
for m in modules:
    tag = " (ERROR)" if "error" in m else ""
    lines.append(f"- {m['module']}{tag} — {m.get('file','<no-file>')}")
(out_dir / "README.md").write_text("\n".join(lines))
print(f"Wrote: {out_dir}")
