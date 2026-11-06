"""Apply the preregistered decision rules (preregistration.json) to the logged results."""

from __future__ import annotations


def _rule(ci_lo: float, ci_hi: float, bound: float, higher_is_support: bool = True) -> str:
    if higher_is_support:
        if ci_lo >= bound:
            return "SUPPORTED"
        if ci_hi < bound:
            return "REFUTED"
    else:
        if ci_hi < bound:
            return "SUPPORTED"
        if ci_lo >= bound:
            return "REFUTED"
    return "INCONCLUSIVE"


def verdict(portfolio: dict, bench: dict, elec: dict) -> dict:
    r_fa, rr = portfolio["H1"]["R_FA"], portfolio["H1"]["RR"]
    v_fa = _rule(r_fa["ci_lo"], r_fa["ci_hi"], 0.90)
    v_rr = _rule(rr["ci_lo"], rr["ci_hi"], 0.80)
    if "REFUTED" in (v_fa, v_rr):
        h1 = "REFUTED"
    elif v_fa == v_rr == "SUPPORTED":
        h1 = "SUPPORTED"
    else:
        h1 = "INCONCLUSIVE"
    gap = bench["H2_psi_fpr_gap"]
    kap = elec["H3_kappa"]
    return {
        "hypothesis": "H1",
        "verdict": h1,
        "H1": {"verdict": h1, "R_FA": r_fa, "R_FA_rule": v_fa, "RR": rr, "RR_rule": v_rr,
               "thresholds": {"R_FA": 0.90, "RR": 0.80}},
        "H2": {"verdict": _rule(gap["ci_lo"], gap["ci_hi"], 0.10), "psi_fpr_gap": gap},
        "H3": {"verdict": _rule(kap["ci_lo"], kap["ci_hi"], 0.4, higher_is_support=False), "kappa": kap},
        "note": "Decision rules as registered in preregistration.json; drift in H1 and H2 is simulated.",
    }
