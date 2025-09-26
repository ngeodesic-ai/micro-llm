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

"""
micro_lm.cli.defi_audit_bench

Usage:
  python -m micro_lm.cli.defi_audit_bench [args identical to benches.audit_bench]

This is a thin shim that forwards to the domain bench:
micro_lm.domains.defi.benches.audit_bench:main
"""
from __future__ import annotations

def main():
    from micro_lm.domains.defi.benches.audit_bench import main as _main
    _main()

if __name__ == "__main__":
    main()
