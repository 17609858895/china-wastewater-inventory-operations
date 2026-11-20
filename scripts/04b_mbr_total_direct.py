# -*- coding: utf-8 -*-
"""Membrane bioreactor contrasts with and without adjustment for effluent composition. Trimming produces different overlap samples."""
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, SEED, load_table, design_matrix

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "results" / "04_causal"
OUT.mkdir(parents=True, exist_ok=True)
NJ = 4
d = load_table()
d = d[(d.year >= 2015) & (d.f_physchem == 0)].reset_index(drop=True)
BASE_FLAGS = ["f_a2o", "f_ao", "f_od", "f_sbr", "f_biofilm", "f_anaer", "f_natural"]
CATS = ["province", "region", "facility", "pathway"]
y = d["log_ei"].values; t = d["f_mbr"].values.astype(int); g = d["group_id"].values
print(f"n={len(d)}, MBR records={t.sum()} ({t.mean():.1%}), groups={len(np.unique(g))}")


def reg():
    return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, n_jobs=NJ, random_state=SEED, verbose=-1)


def clf():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=NJ, random_state=SEED, verbose=-1)


def cluster_se(psi, groups):
    s = pd.Series(psi - psi.mean()).groupby(groups).sum().values
    return float(np.sqrt((s ** 2).sum()) / len(psi))


rows = []
PS = {}
for label, nums in [("Total effect (pre-treatment controls)", ["log_vol", "cod_in", "tn_in", "year"]),
                    ("Controlled direct effect (+ effluent COD, TN)", ["log_vol", "cod_in", "tn_in", "year", "cod_eff", "tn_eff"])]:
    W = design_matrix(d, cats=CATS, nums=nums, flags=BASE_FLAGS)[0].values.astype(np.float32)
    e = np.zeros(len(d)); m1 = np.zeros(len(d)); m0 = np.zeros(len(d)); yh = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(W, groups=g):
        e[te] = clf().fit(W[tr], t[tr]).predict_proba(W[te])[:, 1]
        yh[te] = reg().fit(W[tr], y[tr]).predict(W[te])
        a1, a0 = tr[t[tr] == 1], tr[t[tr] == 0]
        m1[te] = reg().fit(W[a1], y[a1]).predict(W[te])
        m0[te] = reg().fit(W[a0], y[a0]).predict(W[te])
    # PLR
    tr_, yr_ = t - e, y - yh
    th = (tr_ * yr_).sum() / (tr_ ** 2).sum()
    psi = tr_ * (yr_ - th * tr_)
    se = float(np.sqrt((pd.Series(psi).groupby(g).sum().values ** 2).sum()) / (tr_ ** 2).sum())
    rows.append(dict(estimand=label, estimator="PLR-DML", theta=th, se=se, n=len(d)))
    # AIPW with overlap trimming
    for lo in [0.01, 0.05]:
        keep = (e > lo) & (e < 1 - lo)
        ee = np.clip(e[keep], lo, 1 - lo)
        ps = m1[keep] - m0[keep] + t[keep] * (y[keep] - m1[keep]) / ee - (1 - t[keep]) * (y[keep] - m0[keep]) / (1 - ee)
        rows.append(dict(estimand=label, estimator=f"AIPW (trim {lo})", theta=ps.mean(), se=cluster_se(ps, g[keep]),
                         n=int(keep.sum()), n_treated=int(t[keep].sum())))
        if lo == 0.05:
            d["_psi"] = np.nan; d.loc[keep, "_psi"] = ps
            d["scale_class"] = pd.cut(d["q_m3d"], [0, 1e3, 1e4, 5e4, 1e9], labels=["<1k", "1k-10k", "10k-50k", ">50k"])
            for lv, s in d[keep].groupby("scale_class", observed=True):
                rows.append(dict(estimand=label, estimator=f"AIPW by scale {lv}", theta=s["_psi"].mean(),
                                 se=cluster_se(s["_psi"].values, s["group_id"].values), n=len(s), n_treated=int(s.f_mbr.sum())))
    PS["total" if label.startswith("Total") else "direct"] = e.copy()
    print(f"\n[{label}] propensity quantiles treated: {np.quantile(e[t==1], [.05,.5,.95]).round(3)}, "
          f"control: {np.quantile(e[t==0], [.05,.5,.95]).round(4)}; AUC-ish separation: mean e|T=1 {e[t==1].mean():.3f} vs e|T=0 {e[t==0].mean():.4f}")
R = pd.DataFrame(rows)
R["lo"] = R.theta - 1.96 * R.se; R["hi"] = R.theta + 1.96 * R.se
R["pct"] = (10 ** R.theta - 1) * 100; R["pct_lo"] = (10 ** R.lo - 1) * 100; R["pct_hi"] = (10 ** R.hi - 1) * 100
naive = y[t == 1].mean() - y[t == 0].mean()
print(f"\nnaive difference in means: {naive:+.4f} log10 ({(10**naive-1)*100:+.1f}%)")
pd.set_option("display.width", 220)
print(R.round(4).to_string(index=False))
R.to_csv(OUT / "mbr_total_direct.csv", index=False)
np.savez_compressed(OUT / "mbr_propensity.npz", e_total=PS["total"], e_direct=PS["direct"], t=t, q=d["q_m3d"].values)
print("\nMBR effluent quality vs others (median): COD_eff", d.groupby("f_mbr").cod_eff.median().round(1).to_dict(),
      "TN_eff", d.groupby("f_mbr").tn_eff.median().round(1).to_dict(), "| size m3/d", d.groupby("f_mbr").q_m3d.median().round(0).to_dict())
