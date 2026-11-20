# -*- coding: utf-8 -*-
"""07 | 变化归因（"机器学习版 LMDI"）：把两个年份之间平均 log 电耗强度的变化，
        用特征组 Shapley 值精确分解为 规模 / 进水 / 出水 / 工艺 / 区位去向 / 时间残差 的贡献。
  v(S) = E_{(x0,x1)} f(x1[S], x0[~S])，x0 来自基准年、x1 来自目标年（随机配对，干预式）。
  6 组 -> 64 个联盟，逐一精确计算；重复配对给出 Monte-Carlo 标准差。
"""
from pathlib import Path
from itertools import combinations
from math import factorial
import sys, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, SEED, load_table, design_matrix

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "results" / "07_change_attribution"
OUT.mkdir(parents=True, exist_ok=True)
NJ = 4


def fit(X, y):
    return LGBMRegressor(n_estimators=350, learning_rate=0.06, num_leaves=31, min_child_samples=60, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, n_jobs=NJ, random_state=SEED,
                         verbose=-1).fit(X, y)


def group_cols(columns, nums_by_group):
    grp = {}
    for i, c in enumerate(columns):
        if c in nums_by_group:
            key = nums_by_group[c]
        elif c.startswith("f_") or c.startswith(("proc_major_", "proc_sub1_", "proc_sub2_", "tech_")):
            key = "Technology"
        elif c.startswith(("province_", "region_", "pathway_", "facility_")):
            key = "Location & pathway"
        else:
            raise KeyError(c)
        grp.setdefault(key, []).append(i)
    return grp


def decompose(d, nums, nums_by_group, y0, y1, label, n_pairs=3000, reps=3):
    Xdf, _ = design_matrix(d, nums=nums)
    X = Xdf.values.astype(np.float32)
    f = fit(X, d["log_ei"].values)
    grp = group_cols(list(Xdf.columns), nums_by_group)
    names = list(grp.keys())
    i0, i1 = np.where(d.year.values == y0)[0], np.where(d.year.values == y1)[0]
    obs = d.log_ei.values[i1].mean() - d.log_ei.values[i0].mean()
    mod = f.predict(X[i1]).mean() - f.predict(X[i0]).mean()
    rng = np.random.default_rng(SEED)
    phis = []
    for _ in range(reps):
        a, b = X[rng.choice(i0, n_pairs)], X[rng.choice(i1, n_pairs)]
        v = {}
        for r in range(len(names) + 1):
            for S in combinations(range(len(names)), r):
                Z = a.copy()
                for j in S:
                    Z[:, grp[names[j]]] = b[:, grp[names[j]]]
                v[S] = float(f.predict(Z).mean())
        n = len(names)
        phi = np.zeros(n)
        for j in range(n):
            others = [k for k in range(n) if k != j]
            for r in range(n):
                for S in combinations(others, r):
                    w = factorial(r) * factorial(n - r - 1) / factorial(n)
                    phi[j] += w * (v[tuple(sorted(S + (j,)))] - v[S])
        phis.append(phi)
    phis = np.array(phis)
    res = pd.DataFrame({"group": names, "phi_log10": phis.mean(0), "mc_sd": phis.std(0)})
    res["pct_points"] = (10 ** res.phi_log10 - 1) * 100
    res["share_of_change"] = res.phi_log10 / res.phi_log10.sum()
    res["period"] = label
    print(f"\n[{label}] observed change in mean log10 ei = {obs:+.4f} ({(10**obs-1)*100:+.1f}%), model-implied = {mod:+.4f}, "
          f"sum of Shapley = {res.phi_log10.sum():+.4f}")
    print(res.round(4).to_string(index=False))
    return res, dict(period=label, observed=obs, model=mod, shapley_sum=float(res.phi_log10.sum()))


d = load_table()
out, meta = [], []
# COD-only feature set, valid for all years
g_cod = {"log_vol": "Scale", "cod_in": "Influent strength", "cod_eff": "Effluent quality", "year": "Time (unexplained)"}
for y0, y1 in [(2006, 2015), (2015, 2019), (2006, 2019)]:
    r, m = decompose(d, ["log_vol", "cod_in", "cod_eff", "year"], g_cod, y0, y1, f"{y0}->{y1} (COD features, all years)")
    out.append(r); meta.append(m)
# COD + TN feature set, only the years where TN is genuinely reported
d15 = d[d.year >= 2015].reset_index(drop=True)
g_tn = {"log_vol": "Scale", "cod_in": "Influent strength", "tn_in": "Influent strength", "cod_eff": "Effluent quality",
        "tn_eff": "Effluent quality", "year": "Time (unexplained)"}
r, m = decompose(d15, ["log_vol", "cod_in", "tn_in", "cod_eff", "tn_eff", "year"], g_tn, 2015, 2019, "2015->2019 (COD+TN features, 2015-19 model)")
out.append(r); meta.append(m)
pd.concat(out).to_csv(OUT / "shapley_change.csv", index=False)
pd.DataFrame(meta).to_csv(OUT / "shapley_change_meta.csv", index=False)

# composition check: is the post-2015 rise a within-sample change or an entry of new (small) plants?
print("\n=== composition: records by scale class and year (share), and median ei within class ===")
d["scale_class"] = pd.cut(d["q_m3d"], [0, 1e3, 1e4, 5e4, 1e5, 1e9], labels=["<1k", "1k-10k", "10k-50k", "50k-100k", ">100k"])
print(pd.crosstab(d.year, d.scale_class, normalize="index").round(3).loc[[2006, 2010, 2015, 2017, 2019]].to_string())
print(d.pivot_table(index="year", columns="scale_class", values="ei", aggfunc="median", observed=True).round(3).loc[[2006, 2010, 2015, 2017, 2019]].to_string())
vw = d.groupby("year").apply(lambda s: s.elec_1e4kwh.sum() / s.vol_1e4m3.sum())
print("\nvolume-weighted national intensity (kWh/m3):", vw.round(3).to_dict())
