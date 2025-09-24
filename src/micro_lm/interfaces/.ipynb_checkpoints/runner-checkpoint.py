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

from typing import Protocol, Dict, Any, Optional, TypedDict

class RunnerOut(TypedDict, total=False):
    plan: Dict[str, Any]
    verify: Dict[str, Any]
    flags: Dict[str, Any]
    aux: Dict[str, Any]
    label: Optional[str]   # if your pipeline sets it
    score: Optional[float]
    reason: Optional[str]

class RunMicro(Protocol):
    def __call__(
        self,
        domain: str,
        prompt: str,
        *,
        context: Dict[str, Any],
        policy: Dict[str, Any],
        rails: str,
        T: int
    ) -> RunnerOut: ...

# tiny adapter, so your code can be typed but still import dynamically
def bind_run_micro(run_micro_obj) -> RunMicro:
    def _runner(domain, prompt, *, context, policy, rails, T):
        return run_micro_obj(domain, prompt, context=context, policy=policy, rails=rails, T=T)
    return _runner
