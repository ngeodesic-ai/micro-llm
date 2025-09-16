import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
src = root / "src"
if src.exists():
    sys.path.insert(0, str(src))