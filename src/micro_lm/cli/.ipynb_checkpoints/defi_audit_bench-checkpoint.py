
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
