"""
roi.py — financial justification for the solar + battery investment.

Turns the daily savings the optimiser produces into the numbers a real energy
project is approved on: annual savings, simple payback, NPV and IRR.

No external deps — NPV/IRR are computed directly so this runs anywhere.
"""

from dataclasses import dataclass


@dataclass
class ROIResult:
    capex: float            # total upfront cost ($)
    annual_savings: float   # net cash saved per year ($)
    payback_years: float    # simple payback (capex / annual savings)
    npv: float              # net present value over the horizon ($)
    irr_pct: float          # internal rate of return (%), or nan if none
    horizon_years: int
    discount_rate_pct: float


def _npv(rate: float, cashflows: list) -> float:
    """NPV of cashflows[0..N] where cashflows[0] is t=0 (usually negative capex)."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def _irr(cashflows: list, lo=-0.9, hi=1.0, tol=1e-6, max_iter=200):
    """Internal rate of return via bisection. Returns None if no sign change."""
    f_lo, f_hi = _npv(lo, cashflows), _npv(hi, cashflows)
    if f_lo * f_hi > 0:          # no root bracketed -> IRR undefined
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = _npv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def compute_roi(annual_savings: float,
                solar_capex: float,
                battery_capex: float,
                horizon_years: int = 15,
                discount_rate_pct: float = 8.0,
                opex_pct: float = 1.5) -> ROIResult:
    """
    annual_savings   : $ saved per year (e.g. daily optimiser savings * 365)
    solar_capex      : upfront cost of the solar plant ($)
    battery_capex    : upfront cost of the battery ($)
    horizon_years    : project lifetime to evaluate
    discount_rate_pct: discount rate for NPV / hurdle for IRR
    opex_pct         : annual O&M as % of capex (subtracted from savings)
    """
    capex = solar_capex + battery_capex
    opex = capex * (opex_pct / 100.0)
    net_annual = annual_savings - opex

    payback = capex / net_annual if net_annual > 0 else float("inf")

    rate = discount_rate_pct / 100.0
    cashflows = [-capex] + [net_annual] * horizon_years
    npv = _npv(rate, cashflows)
    irr = _irr(cashflows)
    irr_pct = irr * 100 if irr is not None else float("nan")

    return ROIResult(
        capex=capex, annual_savings=net_annual, payback_years=payback,
        npv=npv, irr_pct=irr_pct, horizon_years=horizon_years,
        discount_rate_pct=discount_rate_pct,
    )
