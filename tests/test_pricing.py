"""Tests for the TOU price and carbon intensity proxies."""
import pandas as pd

from battery import tou_price, carbon_intensity


def test_tou_price_peak_and_offpeak():
    idx = pd.date_range("2026-06-01", periods=48, freq="h")  # Mon-Tue
    prices = tou_price(idx)
    assert len(prices) == len(idx)
    for ts, p in zip(idx, prices):
        is_peak = (8 <= ts.hour < 20) and (ts.dayofweek < 5)
        assert p == (80.0 if is_peak else 40.0)


def test_tou_weekend_is_all_offpeak():
    # 2026-06-06 is a Saturday
    idx = pd.date_range("2026-06-06", periods=24, freq="h")
    assert all(p == 40.0 for p in tou_price(idx))


def test_carbon_intensity_shape_and_range():
    idx = pd.date_range("2026-06-01", periods=24, freq="h")
    carbon = carbon_intensity(idx)
    assert len(carbon) == 24
    assert all(200 <= c <= 800 for c in carbon)


def test_carbon_cleaner_midday_than_evening():
    """Duck curve: midday (solar) must be cleaner than the evening peak."""
    idx = pd.date_range("2026-06-01", periods=24, freq="h")
    carbon = carbon_intensity(idx)
    midday = carbon[12]      # 12:00
    evening = carbon[19]     # 19:00
    assert midday < evening
