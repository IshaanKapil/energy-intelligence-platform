"""
battery.py — battery dispatch optimization with Google OR-Tools.

THE INTERVIEW CENTERPIECE. Be able to explain every part of this LP.

PROBLEM (per hour t over a 24h horizon):
  Given forecast load[t], solar[t], price[t], decide how to run a battery and
  draw grid power to MINIMISE total electricity cost.

DECISION VARIABLES (per hour t):
  grid[t]      >= 0   grid power imported (MW)
  charge[t]    >= 0   power used to charge the battery (MW)
  discharge[t] >= 0   power delivered from the battery (MW)
  soc[t]       in [soc_min, soc_max]   battery state of charge (MWh)

CONSTRAINTS:
  (1) Power balance:  solar[t] + grid[t] + discharge[t] - charge[t] = load[t]
  (2) SOC dynamics:   soc[t] = soc[t-1] + charge[t]*eff_c - discharge[t]/eff_d
  (3) Rate limits:    charge[t], discharge[t] <= max_power
  (4) SOC bounds:     soc_min <= soc[t] <= soc_max
  (5) End condition:  soc[last] >= soc_start  (don't drain the battery for free)

OBJECTIVE:
  minimise  sum_t  grid[t] * price[t]

This is a Linear Program -> solved exactly and fast with OR-Tools (GLOP).
"""

from dataclasses import dataclass, field
from typing import List

from ortools.linear_solver import pywraplp


@dataclass
class BatteryConfig:
    capacity_mwh: float = 40.0     # usable energy capacity
    max_power_mw: float = 10.0     # max charge/discharge rate
    eff_charge: float = 0.95       # charging efficiency
    eff_discharge: float = 0.95    # discharging efficiency
    soc_start_frac: float = 0.5    # start at 50% full
    soc_min_frac: float = 0.1
    soc_max_frac: float = 1.0


@dataclass
class DispatchResult:
    grid: List[float] = field(default_factory=list)
    charge: List[float] = field(default_factory=list)
    discharge: List[float] = field(default_factory=list)
    soc: List[float] = field(default_factory=list)
    optimized_cost: float = 0.0
    baseline_cost: float = 0.0     # cost with NO battery
    savings: float = 0.0
    savings_pct: float = 0.0
    status: str = ""


def optimize_battery(load, solar, price, cfg: BatteryConfig = BatteryConfig()) -> DispatchResult:
    T = len(load)
    assert len(solar) == T == len(price), "load/solar/price must be same length"

    solver = pywraplp.Solver.CreateSolver("GLOP")  # linear solver
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver unavailable")

    INF = solver.infinity()
    soc_min = cfg.soc_min_frac * cfg.capacity_mwh
    soc_max = cfg.soc_max_frac * cfg.capacity_mwh
    soc_start = cfg.soc_start_frac * cfg.capacity_mwh

    grid = [solver.NumVar(0, INF, f"grid_{t}") for t in range(T)]
    chg  = [solver.NumVar(0, cfg.max_power_mw, f"chg_{t}") for t in range(T)]
    dis  = [solver.NumVar(0, cfg.max_power_mw, f"dis_{t}") for t in range(T)]
    soc  = [solver.NumVar(soc_min, soc_max, f"soc_{t}") for t in range(T)]

    for t in range(T):
        # (1) power balance: supply meets demand exactly
        solver.Add(solar[t] + grid[t] + dis[t] - chg[t] == load[t])
        # (2) SOC dynamics
        prev = soc_start if t == 0 else soc[t - 1]
        solver.Add(soc[t] == prev + chg[t] * cfg.eff_charge - dis[t] / cfg.eff_discharge)
    # (5) don't end emptier than we started (fair comparison across days)
    solver.Add(soc[T - 1] >= soc_start)

    # objective: minimise grid energy cost
    solver.Minimize(solver.Sum(grid[t] * price[t] for t in range(T)))

    status = solver.Solve()
    status_name = {pywraplp.Solver.OPTIMAL: "OPTIMAL",
                   pywraplp.Solver.FEASIBLE: "FEASIBLE"}.get(status, "NO_SOLUTION")

    res = DispatchResult(status=status_name)
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return res

    res.grid = [grid[t].solution_value() for t in range(T)]
    res.charge = [chg[t].solution_value() for t in range(T)]
    res.discharge = [dis[t].solution_value() for t in range(T)]
    res.soc = [soc[t].solution_value() for t in range(T)]
    res.optimized_cost = solver.Objective().Value()

    # baseline: no battery -> import whatever load isn't covered by solar
    res.baseline_cost = sum(max(load[t] - solar[t], 0.0) * price[t] for t in range(T))
    res.savings = res.baseline_cost - res.optimized_cost
    res.savings_pct = 100 * res.savings / res.baseline_cost if res.baseline_cost else 0.0
    return res


def tou_price(index) -> list:
    """Time-of-use electricity price proxy ($/MWh).
    Peak (8am-8pm weekdays) = $80, off-peak = $40.
    Realistic stand-in when no market price data is available.
    """
    prices = []
    for ts in index:
        is_peak = (8 <= ts.hour < 20) and (ts.dayofweek < 5)
        prices.append(80.0 if is_peak else 40.0)
    return prices


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[2]
    df = pd.read_csv(ROOT / "data" / "energy_dataset_real.csv",
                     parse_dates=["datetime"]).set_index("datetime").iloc[-24:]
    price = tou_price(df.index)
    res = optimize_battery(df.load_mw.tolist(), df.solar_mw.tolist(),
                           price, BatteryConfig())

    print(f"status            : {res.status}")
    print(f"baseline cost     : {res.baseline_cost:,.0f}")
    print(f"optimized cost    : {res.optimized_cost:,.0f}")
    print(f"savings           : {res.savings:,.0f}  ({res.savings_pct:.1f}%)")

    REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)
    sched = pd.DataFrame({
        "datetime": df.index, "load": df.load_mw.values, "solar": df.solar_mw.values,
        "price": price, "grid": res.grid, "charge": res.charge,
        "discharge": res.discharge, "soc": res.soc,
    })
    sched.to_csv(REPORTS / "battery_schedule.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    a1.plot(df.index, df.load_mw, "k-", label="Load")
    a1.plot(df.index, df.solar_mw, color="#f59e0b", label="Solar")
    a1.bar(df.index, res.charge, width=0.03, color="#16a34a", label="Charge")
    a1.bar(df.index, [-d for d in res.discharge], width=0.03, color="#dc2626", label="Discharge")
    a1.set_ylabel("MW"); a1.legend(ncol=4, fontsize=8); a1.grid(alpha=0.3)
    a1.set_title(f"Battery dispatch — saves {res.savings_pct:.1f}% vs grid-only")
    a2b = a2.twinx()
    a2.plot(df.index, res.soc, color="#7c3aed", label="SOC (MWh)")
    a2b.plot(df.index, df.price, color="#9ca3af", ls=":", label="Price")
    a2.set_ylabel("SOC (MWh)"); a2b.set_ylabel("Price")
    a2.grid(alpha=0.3); a2.legend(loc="upper left", fontsize=8); a2b.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "battery_dispatch.png", dpi=120)
    print("saved reports/battery_dispatch.png + battery_schedule.csv")
