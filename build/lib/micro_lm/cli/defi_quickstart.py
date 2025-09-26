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

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json, argparse, os
from micro_lm.core.runner import run_micro  # public in module map
from micro_lm.interfaces.defi_prompt import run_defi  # public in module map
from micro_lm.domains.defi.verify_local import verify_action_local  # package map shows it's exported
import hashlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--policy", default="")
    ap.add_argument("--context", default="")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--use_wdd", action="store_true")
    ap.add_argument("--pca_prior", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    policy = json.loads(args.policy) if args.policy else None
    context = json.loads(args.context) if args.context else None
    out = run_defi(
        args.prompt,
        policy=policy,
        context=context,
        rails=args.rails,
        T=args.T,
        use_wdd=args.use_wdd,
        pca_prior=args.pca_prior,
        profile=args.profile,
        verbose=args.verbose,
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

