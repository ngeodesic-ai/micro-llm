# src/micro_lm/compat/legacy_imports.py
import sys, importlib, __main__

ALIASES = {
    "encoders": "micro_lm.domains.defi.benches.encoders",
    "__main__": "micro_lm.domains.defi.benches.encoders",  # allow __main__.SBERTEncoder
}

def _export_to_main(mod, names=("SBERTEncoder","SbertEncoder","EmbedVectorizer","SBERTVectorizer")):
    for n in names:
        if hasattr(mod, n) and not hasattr(__main__, n):
            setattr(__main__, n, getattr(mod, n))

def install():
    for legacy, target in ALIASES.items():
        try:
            mod = importlib.import_module(target)
            sys.modules.setdefault(legacy, mod)
            _export_to_main(mod)  # <- ensures __main__.SBERTEncoder exists for unpickling
        except Exception:
            pass
