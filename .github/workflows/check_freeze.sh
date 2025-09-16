#!/usr/bin/env bash
set -euo pipefail
diff <(find scripts/milestones -type f -print0 | sort -z | xargs -0 shasum -a 256) .milestones.freeze.sha256

