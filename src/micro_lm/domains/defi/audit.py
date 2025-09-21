# Legacy shim so runner + benches that import micro_lm.domains.defi.audit keep working.
# We delegate to the new WDD backend by default on this branch.
from .audit_wdd import wdd_defi_audit as defi_audit

__all__ = ["defi_audit"]
