import os
from micro_lm.cli.defi_quickstart import quickstart

def test_wdd_detector_activity_deposit(monkeypatch):
    """
    Detector mode: ensure WDD runs and populates the summary fields.
    This does NOT require a PCA prior file.
    """
    # turn on harness debug (stderr) but we don't assert on it
    monkeypatch.setenv("MICRO_LM_WDD_DEBUG", "1")

    out = quickstart(
        "deposit 10 ETH into aave",
        rails="stage11",
        use_wdd=True,
        verbose=True
    )

    # # Plan should reflect deposit
    assert out["plan"]["sequence"][:1] == ["deposit_asset"]
    assert out["wdd_summary"]["decision"] == "PASS"
    assert out["wdd_summary"]["which_prior"] == "deposit(L-5)"

# def test_wdd_family_mode_returns_order():
#     """
#     Family mode: WDD supplies the execution order and verify reason.
#     """
#     out = quickstart(
#         "swap 2 ETH for USDC",
#         rails="stage11",
#         policy={
#             "audit": {
#                 "backend": "wdd",
#                 "mode": "family",
#                 "K": 12,
#                 "template_width": 64,
#                 "z_abs": 0.55,
#                 "keep_frac": 0.70,
#             }
#         },
#     )

#     # Family mode should stamp verify.reason and set the family flag
#     assert (out.get("verify") or {}).get("reason") == "wdd:family:defi"
#     assert (out.get("flags") or {}).get("wdd_family") is True

#     # The plan's sequence should come from aux.wdd.order
#     fam = (out.get("aux") or {}).get("wdd") or {}
#     order = fam.get("order") or []
#     assert isinstance(order, list) and len(order) >= 1
#     assert out["plan"]["sequence"] == order
