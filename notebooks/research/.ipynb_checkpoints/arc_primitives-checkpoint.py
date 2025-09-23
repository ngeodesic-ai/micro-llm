import json, ast
import numpy as np

def flip_h(grid):
    g = np.asarray(grid)
    return np.fliplr(g)

def flip_v(grid):
    g = np.asarray(grid)
    return np.flipud(g)

def rot90(grid, k=1, direction="cw"):
    g = np.asarray(grid)
    if direction.lower() == "cw":
        return np.rot90(g, k=(4 - (k % 4)) % 4)
    else:
        return np.rot90(g, k=(k % 4))

def rotate(grid, degrees=90, direction="cw"):
    deg = ((degrees % 360) + 360) % 360
    if deg == 0:
        return np.asarray(grid).copy()
    if deg == 90:
        return rot90(grid, k=1, direction=direction)
    if deg == 180:
        return rot90(grid, k=2, direction="ccw")
    if deg == 270:
        return rot90(grid, k=3, direction="ccw" if direction.lower()=="cw" else "cw")
    raise ValueError("rotate: degrees must be one of {0,90,180,270}")

PRIMITIVES = {
    "flip_h": flip_h,
    "flip_v": flip_v,
    "rotate": lambda g: rotate(g, 90, "cw"),
    "rot90":  lambda g: rotate(g, 90, "cw"),
    "rot90_cw":  lambda g: rotate(g, 90, "cw"),
    "rot90_ccw": lambda g: rotate(g, 90, "ccw"),
    "rot180": lambda g: rotate(g, 180, "cw"),
    "rot270": lambda g: rotate(g, 270, "cw"),
}

def parse_grid(text):
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=int)
    s = str(text).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(s)
            return np.asarray(obj, dtype=int)
        except Exception:
            pass
    rows = [r for r in s.replace("\n",";").split(";") if r.strip()]
    parsed_rows = []
    for r in rows:
        cells = [c for c in r.replace(",", " ").split(" ") if c.strip()]
        parsed_rows.append([int(c) for c in cells])
    return np.asarray(parsed_rows, dtype=int)

def apply_label(grid, label):
    if label not in PRIMITIVES:
        raise KeyError(f"Unknown label '{label}'. Known: {sorted(PRIMITIVES.keys())}")
    return PRIMITIVES[label](grid)