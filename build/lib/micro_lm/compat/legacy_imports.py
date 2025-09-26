"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

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
