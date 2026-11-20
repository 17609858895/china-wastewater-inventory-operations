# -*- coding: utf-8 -*-
"""Grouped quantile regression, conditional electricity reference gaps, and residual persistence."""
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, SEED, load_table, design_matrix

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "results" / "06_frontier"
OUT.mkdir(parents=True, exist_ok=True)
NJ = 8
TAUS = [0.10, 0.25, 0.50, 0.90]
NUMS = ["log_vol", "cod_in", "cod_eff", "year"]
MONO = {"log_vol": -1, "cod_in": 1, "cod_eff": -1}

d = load_table()
Xdf, _ = design_matrix(d, nums=NUMS)
X = Xdf.values.astype(np.float32)
y = d["log_ei"].values
g = d["group_id"].values
mono = [MONO.get(c, 0) for c in Xdf.columns]


def qmodel(tau, constrained):
    # LightGBM 的 quantile 目标不支持单调约束；sklearn HistGradientBoosting 同时支持分位数损失与 monotonic_cst
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_iter=250, learning_rate=0.07,
                                         max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=3.0,
                                         monotonic_cst=mono if constrained else None, random_state=SEED)


def pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


folds = list(GroupKFold(5).split(X, groups=g))
Q = {}
rows = []
for constrained in [False, True]:  # 无约束只算 0.10/0.50 以节省时间
    for tau in (TAUS if constrained else [0.10, 0.50]):
        q = np.zeros(len(y))
        for tr, te in folds:
            q[te] = qmodel(tau, constrained).fit(X[tr], y[tr]).predict(X[te])
        Q[(constrained, tau)] = q
        # split-conformal style calibration constant (computed on OOF residuals): shift so coverage == tau
        shift = float(np.quantile(y - q, tau))
        rows.append(dict(constrained=constrained, tau=tau, pinball=pinball(y, q, tau), coverage=float((y < q).mean()),
                         calib_shift=shift, pinball_calibrated=pinball(y, q + shift, tau)))
        print(rows[-1], flush=True)
cal = pd.DataFrame(rows)
cal.to_csv(OUT / "quantile_quality.csv", index=False)

# quantile crossing check and monotonicity violations of the unconstrained frontier
for c in [False, True]:
    cross = float((Q[(c, 0.10)] > Q[(c, 0.50)]).mean())
    print(f"constrained={c}: share of records with q10 > q50 (crossing) = {cross:.4f}")

# use the constrained, calibrated quantiles
shift = {t: float(cal[(cal.constrained) & (cal.tau == t)].calib_shift.iloc[0]) for t in TAUS}
for t in TAUS:
    d[f"q{int(t*100)}"] = Q[(True, t)] + shift[t]
vol_m3 = d["vol_1e4m3"].values * 1e4
E = d["elec_1e4kwh"].values * 1e4                     # kWh
ef = d["grid_ef"].values                               # kg CO2/kWh
out = []
for label, col in [("to frontier (tau=0.10)", "q10"), ("to lower quartile (tau=0.25)", "q25"), ("to conditional median (tau=0.50)", "q50")]:
    E_ref = vol_m3 * 10 ** d[col].values
    save = np.maximum(E - E_ref, 0.0)
    d[f"save_{col}"] = save
    by = pd.DataFrame({"year": d.year, "E": E, "save": save, "co2": save * ef / 1e9, "co2_tot": E * ef / 1e9}).groupby("year").sum()
    by["save_share"] = by["save"] / by["E"]
    by["scenario"] = label
    out.append(by.reset_index())
    r = by.loc[2019]
    print(f"\n[{label}] 2019: electricity {r.E/1e9:.2f} TWh (complete-case sample), saving {r.save/1e9:.2f} TWh "
          f"({r.save_share:.1%}), CO2 avoided {r.co2:.2f} Mt of {r.co2_tot:.2f} Mt")
sav = pd.concat(out)
sav.to_csv(OUT / "saving_potential_by_year.csv", index=False)

d["gap_log"] = y - d["q10"].values                     # distance to frontier in log10 units
d["eff_ratio"] = 10 ** (d["q10"].values - y)           # frontier / actual (<=1 means inefficient)
print("\nefficiency ratio (frontier/actual) quantiles:", d["eff_ratio"].quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
d["scale_class"] = pd.cut(d["q_m3d"], [0, 1e3, 1e4, 5e4, 1e5, 1e9], labels=["<1k", "1k-10k", "10k-50k", "50k-100k", ">100k"])
y19 = d[d.year == 2019]
for col in ["scale_class", "tech", "region"]:
    t = y19.groupby(col, observed=True).apply(lambda s: pd.Series({
        "n": len(s), "elec_TWh": s.elec_1e4kwh.sum() / 1e5, "median_gap_pct": (10 ** s.gap_log.median() - 1) * 100,
        "save_q10_TWh": s.save_q10.sum() / 1e9, "save_q50_TWh": s.save_q50.sum() / 1e9,
        "save_q10_share": s.save_q10.sum() / (s.elec_1e4kwh.sum() * 1e4)}))
    t.to_csv(OUT / f"gap_2019_by_{col}.csv")
    print(f"\n2019 by {col}:\n", t.round(3).to_string())

# persistence of the efficiency residual along reconstructed trajectories
full = pd.read_parquet(ROOT / "results" / "cache" / "model_table.parquet").sort_values("no").reset_index(drop=True)
L = pd.read_parquet(ROOT / "results" / "02c_linkage" / "links.parquet")
res = pd.Series((y - d["q50"].values), index=d["no"].values)
ra = res.reindex(full.loc[L.i.values, "no"].values).values
rb = res.reindex(full.loc[L.j.values, "no"].values).values
ok = np.isfinite(ra) & np.isfinite(rb)
changed = ok & (np.abs(full.loc[L.i.values, 'log_ei'].values - full.loc[L.j.values, 'log_ei'].values) > 1e-9)
print(f"\npersistence of efficiency residual (log ei - q50) between consecutive linked years: "
      f"r = {np.corrcoef(ra[ok], rb[ok])[0,1]:.3f} (n={ok.sum()}); excluding exact carry-overs r = "
      f"{np.corrcoef(ra[changed], rb[changed])[0,1]:.3f} (n={changed.sum()}); fills only r = "
      f"{np.corrcoef(ra[ok & (L.how.values==3)], rb[ok & (L.how.values==3)])[0,1]:.3f}")
d["q10_unc"] = Q[(False, 0.10)] + float(cal[(~cal.constrained) & (cal.tau == 0.10)].calib_shift.iloc[0])
d["q10_raw"] = Q[(True, 0.10)]
d[["no", "year", "province", "region", "tech", "q_m3d", "log_vol", "cod_in", "cod_eff", "ei", "log_ei", "q10", "q25", "q50", "q90",
   "q10_unc", "q10_raw", "gap_log", "eff_ratio", "save_q10", "save_q25", "save_q50", "grid_ef", "elec_1e4kwh"]].to_parquet(OUT / "frontier_oof.parquet", index=False)
print("saved", OUT)
