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

import argparse, json
from micro_lm.core.runner import run_micro

def main(argv=None):
    ap = argparse.ArgumentParser("micro-lm")
    ap.add_argument("domain", choices=["defi", "arc"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--context", default="{}")
    ap.add_argument("--policy", default="{}")
    ap.add_argument("--backend", default="sbert")
    args = ap.parse_args(argv)

    out = run_micro(
        args.domain,
        args.prompt,
        context=json.loads(args.context),
        policy=json.loads(args.policy),
        rails=args.rails,
        T=args.T,
        backend=args.backend,
    )
    print(json.dumps(out, indent=2))
