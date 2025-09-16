def test_micro_lm_import_does_not_pull_domains():
    # Ensure importing micro_lm doesn't auto-import any domain modules.
    import sys
    sys.modules.pop("micro_lm", None)
    before = set(sys.modules.keys())
    import micro_lm  # noqa: F401
    after = set(sys.modules.keys())
    newly = after - before
    assert not any(m.startswith("micro_lm.domains") for m in newly)
