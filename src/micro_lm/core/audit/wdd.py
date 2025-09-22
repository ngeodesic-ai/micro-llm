from pathlib import Path
import json
from micro_lm.core.utils import timed

def wdd_orchestrate(..., profile_dir: Path | None = None, ...):
    profile = []
    with timed("shortlist", profile):
        # shortlist work
        ...
    with timed("traces", profile):
        # vectors/paths
        ...
    with timed("parse", profile):
        # detector.parse(...)
        ...
    with timed("order", profile):
        # order/merge/etc.
        ...

    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "profile.json").write_text(json.dumps(profile, indent=2))

    return result
