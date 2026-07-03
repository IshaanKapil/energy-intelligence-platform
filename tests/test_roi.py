"""Tests for the ROI engine (payback / NPV / IRR)."""
import math

from roi import compute_roi, _npv, _irr


def test_payback_is_capex_over_net_savings():
    # No opex -> payback is exactly capex / annual savings.
    r = compute_roi(annual_savings=100_000, solar_capex=600_000,
                    battery_capex=400_000, horizon_years=15, opex_pct=0.0)
    assert r.capex == 1_000_000
    assert r.payback_years == 10.0


def test_positive_project_has_positive_npv_and_irr():
    # Savings large vs capex -> clearly profitable.
    r = compute_roi(annual_savings=300_000, solar_capex=500_000,
                    battery_capex=500_000, horizon_years=15,
                    discount_rate_pct=8.0, opex_pct=1.0)
    assert r.npv > 0
    assert r.irr_pct > 8.0            # clears the hurdle rate


def test_unprofitable_project_has_negative_npv():
    # Tiny savings vs huge capex -> not viable.
    r = compute_roi(annual_savings=50_000, solar_capex=20_000_000,
                    battery_capex=10_000_000, horizon_years=15)
    assert r.npv < 0
    assert r.payback_years > r.horizon_years


def test_npv_zero_at_irr():
    # By definition, discounting at the IRR gives NPV ~ 0.
    cashflows = [-1_000_000] + [150_000] * 15
    irr = _irr(cashflows)
    assert irr is not None
    assert abs(_npv(irr, cashflows)) < 1.0   # NPV ~ 0 when discounted at the IRR


def test_irr_none_when_never_pays_back():
    # All-negative future flows -> no IRR.
    cashflows = [-1_000_000] + [-10_000] * 15
    assert _irr(cashflows) is None
