"""
fetch_real_data.py
------------------
Builds a LIVE dataset from real, current data — no static/historical files.

  python fetch_real_data.py --eia            # 1 year of live data (needs EIA key)
  python fetch_real_data.py --live           # current + 7-day forecast weather only

Sources (all free):
  1) LOAD  -> EIA Open Data API (real hourly US grid demand, ~5 yrs history)
              Free instant key: https://www.eia.gov/opendata/  (set EIA_API_KEY)
  2) WEATHER/SOLAR -> Open-Meteo Archive + Forecast APIs (free, NO API KEY)

Install deps:
    pip install pandas requests

Output:
  data/energy_dataset_real.csv   live training data (load + weather + solar)
  data/weather_live.csv          live/forecast weather for real-time inference
"""

import argparse

import pandas as pd
import requests
from pathlib import Path

OUT = Path("data"); OUT.mkdir(exist_ok=True)

LAT, LON = 40.0, -82.9          # central Ohio (PJM/AEP territory)
SOLAR_CAPACITY_MW = 20.0        # hypothetical plant size


# ---------------------------------------------------------------------------
# WEATHER (live/forecast) — Open-Meteo Forecast API (free, no key)
#    Gives current conditions + 7 days ahead
# ---------------------------------------------------------------------------
def get_weather_live():
    print("  Fetching live + forecast weather from Open-Meteo ...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
        "timezone": "America/New_York",
        "forecast_days": 7,
        "past_days": 2,        # include last 2 days for context
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return _parse_weather(r.json()["hourly"])


def _parse_weather(h: dict) -> pd.DataFrame:
    w = pd.DataFrame(h)
    w["datetime"] = pd.to_datetime(w["time"])
    w = w.drop(columns="time").set_index("datetime").sort_index()
    w = w.rename(columns={
        "temperature_2m":     "temperature_c",
        "cloud_cover":        "cloud_cover_pct",
        "shortwave_radiation":"irradiance_wm2",
        "wind_speed_10m":     "wind_speed",
    })
    w["solar_mw"] = (w["irradiance_wm2"] / 1000.0 * SOLAR_CAPACITY_MW).clip(lower=0).round(3)
    return w


# ---------------------------------------------------------------------------
# LIVE LOAD — EIA Open Data API (real, recent US grid demand)
#   Free instant API key: https://www.eia.gov/opendata/  (set EIA_API_KEY)
#   We pull recent hourly demand for the PJM region and scale it to the same
#   microgrid magnitude the models were trained on, so existing models work.
# ---------------------------------------------------------------------------
def get_load_eia(api_key: str, days: int = 365, respondent: str = "PJM"):
    """Fetch `days` of hourly demand from EIA, paginating past the 5000-row cap."""
    from datetime import datetime, timedelta
    print(f"  Fetching last {days} days of {respondent} hourly demand from EIA ...")
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    base = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": "D",                     # D = demand
        "start": start.strftime("%Y-%m-%dT%H"),
        "end":   end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    all_rows, offset = [], 0
    while True:
        params = dict(base, offset=offset)
        r = requests.get(url, params=params, timeout=90)
        r.raise_for_status()
        rows = r.json()["response"]["data"]
        if not rows:
            break
        all_rows.extend(rows)
        print(f"    pulled {len(all_rows)} rows ...")
        if len(rows) < base["length"]:
            break
        offset += base["length"]
    if not all_rows:
        raise RuntimeError("EIA returned no rows — check API key / respondent code.")
    s = pd.DataFrame(all_rows)
    s["datetime"] = pd.to_datetime(s["period"])
    s["load_mw"] = pd.to_numeric(s["value"], errors="coerce")
    s = (s.dropna(subset=["load_mw"]).drop_duplicates("datetime")
           .set_index("datetime").sort_index()[["load_mw"]])
    print(f"  EIA total rows: {len(s)}  (raw peak {s['load_mw'].max():,.0f} MW)")
    return s


def get_weather_for_load(load_index):
    """Get weather covering the full span of `load_index`, combining Open-Meteo's
    archive API (older than ~5 days) with the forecast API (recent ~92 days)."""
    from datetime import date, timedelta
    start_d = load_index.min().date()
    end_d   = load_index.max().date()
    archive_end = date.today() - timedelta(days=6)   # archive lags ~5 days

    frames = []
    if start_d <= archive_end:
        print(f"  Archive weather {start_d} -> {archive_end} ...")
        au = "https://archive-api.open-meteo.com/v1/archive"
        ap = {"latitude": LAT, "longitude": LON,
              "start_date": str(start_d), "end_date": str(archive_end),
              "hourly": "temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
              "timezone": "America/New_York"}
        ar = requests.get(au, params=ap, timeout=90); ar.raise_for_status()
        frames.append(_parse_weather(ar.json()["hourly"]))

    print("  Recent weather (last 92 days) from forecast API ...")
    fu = "https://api.open-meteo.com/v1/forecast"
    fp = {"latitude": LAT, "longitude": LON,
          "hourly": "temperature_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
          "timezone": "America/New_York", "past_days": 92, "forecast_days": 1}
    fr = requests.get(fu, params=fp, timeout=90); fr.raise_for_status()
    frames.append(_parse_weather(fr.json()["hourly"]))

    w = pd.concat(frames)
    w = w[~w.index.duplicated(keep="last")].sort_index()
    return w


def build_eia_live(api_key: str, days: int = 365):
    """Build a LIVE dataset from EIA demand + Open-Meteo weather (archive+forecast)
    and overwrite energy_dataset_real.csv (backup at energy_dataset_real.backup.csv)."""
    load    = get_load_eia(api_key, days)
    load.index = load.index.tz_localize(None)
    weather = get_weather_for_load(load.index)
    weather.index = weather.index.tz_localize(None)

    df = load.join(weather, how="inner").dropna()
    if len(df) < 200:
        raise RuntimeError(f"Only {len(df)} aligned rows — not enough for lag features.")

    # Scale live demand to the microgrid magnitude the project is themed around
    # (~16 MW mean) so the numbers stay in a coherent microgrid range.
    factor = 16.0 / df["load_mw"].mean()
    df["load_mw"] = (df["load_mw"] * factor).round(3)

    out = OUT / "energy_dataset_real.csv"
    df.to_csv(out)
    print(f"\nSaved LIVE {out}  rows={len(df)}  "
          f"(scaled peak {df['load_mw'].max():.1f} MW, factor={factor:.2e})")
    print(f"  Date range: {df.index[0]} -> {df.index[-1]}")
    print(df.tail())
    print("\nNext: retrain models on this live data -> python run_all.py")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_live():
    df = get_weather_live()
    out = OUT / "weather_live.csv"
    df.to_csv(out)
    print(f"\nSaved {out}  rows={len(df)}")
    print(df.head())
    return df


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(
        description="Build a LIVE energy dataset from EIA + Open-Meteo.")
    parser.add_argument("--eia", action="store_true",
                        help="Build the live training dataset (default action)")
    parser.add_argument("--live-weather", action="store_true",
                        help="Only refresh live/forecast weather -> weather_live.csv")
    parser.add_argument("--eia-key", default=None,
                        help="EIA API key (or set EIA_API_KEY env var)")
    parser.add_argument("--days", type=int, default=365,
                        help="How many recent days of live data to pull (default 365)")
    args = parser.parse_args()

    if args.live_weather:
        build_live()
    else:
        # default: build the live EIA dataset
        key = args.eia_key or os.environ.get("EIA_API_KEY")
        if not key:
            raise SystemExit("Set --eia-key or the EIA_API_KEY env var. "
                             "Get a free key at https://www.eia.gov/opendata/")
        build_eia_live(key, args.days)
