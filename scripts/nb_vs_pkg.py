#!/usr/bin/env python3
import json, pathlib, ast, sys
import nbformat

NB_DIRS = ["notebooks/templates", "."]  # add your notebook dirs
PKG_MAP = pathlib.Path(".artifacts/pkg_map/pkg_modules.json")

def extract_defs_from_ipynb(path):
    nb = nbformat.read(path, as_version=4)
    names = set()
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            try:
                tree = ast.parse(cell.source)
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if not node.name.startswith("_"):
                            names.add(node.name)
            except Exception:
                pass
    return names

# 1) all notebook symbols
nb_names = set()
for d in NB_DIRS:
    for p in pathlib.Path(d).rglob("*.ipynb"):
        nb_names |= extract_defs_from_ipynb(p)

# 2) package public symbols
pkg = json.loads(PKG_MAP.read_text())
pkg_names = set()
for m in pkg:
    if "public" in m and isinstance(m["public"], list):
        for n in m["public"]:
            pkg_names.add(n)

print(json.dumps({
    "only_in_notebooks": sorted(nb_names - pkg_names),
    "only_in_package":   sorted(pkg_names - nb_names),
    "intersection":      sorted(nb_names & pkg_names)[:50],
}, indent=2))
