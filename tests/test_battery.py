"""Tests for the battery dispatch optimizer (OR-Tools LP)."""
import pandas as pd
import pytest

from battery import (optimize_battery, BatteryConfig, tou_price,
                     carbon_intensity, CARBON_PRICE_MODES)


@pytest.fixture
def sample_day():
    """A representative 24h microgrid day."""
    idx = pd.date_range("2026-06-01", periods=24, freq="h")
    load = [14 + 5 * (8 <= t.hour < 22) for t in idx]           # ~14-19 MW
    solar = [max(0.0, (t.hour - 6) * (18 - t.hour)) * 0.13 for t in idx]
    return idx, load, solar


def test_optimizer_returns_optimal(sample_day):
    idx, load, solar = sample_day
    res = optimize_battery(load, solar, tou_price(idx), BatteryConfig())
    assert res.status == "OPTIMAL"


def test_soc_stays_within_capacity(sample_day):
    idx, load, solar = sample_day
    cfg = BatteryConfig(capacity_mwh=40, max_power_mw=10)
    res = optimize_battery(load, solar, tou_price(idx), cfg)
    soc_min = cfg.soc_min_frac * cfg.capacity_mwh
    soc_max = cfg.soc_max_frac * cfg.capacity_mwh
    for s in res.soc:
        assert soc_min - 1e-6 <= s <= soc_max + 1e-6


def test_charge_discharge_within_power_limit(sample_day):
    idx, load, solar = sample_day
    cfg = BatteryConfig(max_power_mw=10)
    res = optimize_battery(load, solar, tou_price(idx), cfg)
    for c, d in zip(res.charge, res.discharge):
        assert -1e-6 <= c <= cfg.max_power_mw + 1e-6
        assert -1e-6 <= d <= cfg.max_power_mw + 1e-6


def test_savings_never_negative_in_cost_mode(sample_day):
    idx, load, solar = sample_day
    # Cost mode: an optimiser can never do WORSE than the no-battery baseline.
    res = optimize_battery(load, solar, tou_price(idx), BatteryConfig())
    assert res.savings >= -1e-6
    assert res.optimized_cost <= res.baseline_cost + 1e-6


def test_power_balance_holds(sample_day):
    idx, load, solar = sample_day
    res = optimize_battery(load, solar, tou_price(idx), BatteryConfig())
    # solar + grid + discharge - charge == load, every hour
    for t in range(24):
        supply = solar[t] + res.grid[t] + res.discharge[t] - res.charge[t]
        assert supply == pytest.approx(load[t], abs=1e-4)


def test_green_mode_cuts_more_carbon_than_cost_mode(sample_day):
    idx, load, solar = sample_day
    carbon = carbon_intensity(idx)
    cost = optimize_battery(load, solar, tou_price(idx), BatteryConfig(),
                            carbon=carbon, carbon_price=CARBON_PRICE_MODES["cost"])
    green = optimize_battery(load, solar, tou_price(idx), BatteryConfig(),
                             carbon=carbon, carbon_price=CARBON_PRICE_MODES["green"])
    # Green mode should avoid at least as much CO2 as cost mode.
    assert green.emissions_saved >= cost.emissions_saved - 1e-6
    # ...but never beat cost mode on pure $ savings.
    assert green.savings <= cost.savings + 1e-6
