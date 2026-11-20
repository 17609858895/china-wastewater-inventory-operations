# -*- coding: utf-8 -*-
"""Cross-fitted partially linear associations between plant attributes and electricity intensity."""
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
NJ = 8
rng = np.random.default_rng(SEED)

d = load_table()
CATS = ["province", "region", "facility", "proc_major", "proc_sub1", "proc_sub2", "pathway", "tech"]
FLAGS = ["f_a2o", "f_ao", "f_od", "f_sbr", "f_mbr", "f_biofilm", "f_anaer", "f_natural", "f_physchem", "f_two_stage"]


def lgb(clf=False):
    kw = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=NJ, random_state=SEED, verbose=-1)
    return LGBMClassifier(**kw) if clf else LGBMRegressor(**kw)


def cross_fit(W, v, groups, clf=False, k=5):
    out = np.zeros(len(v))
    for tr, te in GroupKFold(k).split(W, groups=groups):
        m = lgb(clf).fit(W[tr], v[tr])
        out[te] = m.predict_proba(W[te])[:, 1] if clf else m.predict(W[te])
    return out


def plr(y_res, t_res, groups):
    """theta = sum(t~ y~)/sum(t~^2)，聚类稳健 SE。"""
    theta = float((t_res * y_res).sum() / (t_res ** 2).sum())
    psi = t_res * (y_res - theta * t_res)
    g = pd.Series(psi).groupby(groups).sum().values
    se = float(np.sqrt((g ** 2).sum()) / (t_res ** 2).sum())
    # partial f and robustness value (Cinelli & Hazlett 2020)
    tval = theta / se
    dof = len(np.unique(groups)) - 1
    f = abs(tval) / np.sqrt(dof)
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
    r2_t = 1 - (y_res - theta * t_res).var() / y_res.var()
    return dict(theta=theta, se=se, lo=theta - 1.96 * se, hi=theta + 1.96 * se, t=tval, RV=rv, partial_R2_Y_T=r2_t)


def run(name, data, tcol, tscale, nums, binary=False, drop_flags=()):
    flags = [f for f in FLAGS if f not in drop_flags]
    cats = [c for c in CATS if not (binary and c in ("proc_sub2", "tech", "proc_sub1", "proc_major"))]
    Wdf, _ = design_matrix(data, cats=cats, nums=nums, flags=flags)
    W = Wdf.values.astype(np.float32)
    y = data["log_ei"].values
    t = data[tcol].values.astype(float) / tscale
    g = data["group_id"].values
    y_hat = cross_fit(W, y, g)
    t_hat = cross_fit(W, t, g, clf=binary)
    yr, tr_ = y - y_hat, t - t_hat
    res = plr(yr, tr_, g)
    # naive OLS slope without controls, for contrast
    res["naive_slope"] = float(np.polyfit(t, y, 1)[0])
    res["R2_y_nuis"] = 1 - yr.var() / y.var()
    res["R2_t_nuis"] = 1 - tr_.var() / t.var()
    # placebo: permute residualised treatment within province-year
    key = data["province"].astype(str) + "_" + data["year"].astype(str)
    tp = tr_.copy()
    for _, ix in pd.Series(np.arange(len(tp))).groupby(key.values):
        tp[ix.values] = rng.permutation(tp[ix.values])
    res["placebo_theta"] = plr(yr, tp, g)["theta"]
    res.update(estimand=name, n=len(data), n_groups=int(len(np.unique(g))))
    pct = (10 ** res["theta"] - 1) * 100
    print(f"\n[{name}] n={len(data)}  theta={res['theta']:+.4f} log10 units ({pct:+.1f}% electricity intensity) "
          f"95%CI [{res['lo']:+.4f}, {res['hi']:+.4f}]  naive={res['naive_slope']:+.4f}  RV={res['RV']:.3f}  "
          f"placebo={res['placebo_theta']:+.5f}  R2(y|W)={res['R2_y_nuis']:.3f} R2(t|W)={res['R2_t_nuis']:.3f}")
    return res, yr, tr_


results = []
# sign convention: treatment = -(effluent conc)/scale  => "per 10 mg/L cleaner"
d["t_cod_clean"] = -d["cod_eff"]
d["t_tn_clean"] = -d["tn_eff"]
base_nums = ["log_vol", "cod_in", "year"]

r1, yr1, tr1 = run("Cleaner effluent COD (per -10 mg/L), 2006-2019", d, "t_cod_clean", 10.0, base_nums)
results.append(r1)
d15 = d[d.year >= 2015].reset_index(drop=True)
r1b, _, _ = run("Cleaner effluent COD (per -10 mg/L), 2015-2019", d15, "t_cod_clean", 10.0, base_nums + ["tn_in", "tn_eff"])
results.append(r1b)
r2, yr2, tr2 = run("Cleaner effluent TN (per -5 mg/L), 2015-2019", d15, "t_tn_clean", 5.0, base_nums + ["tn_in", "cod_eff"])
results.append(r2)
r3, yr3, tr3 = run("MBR vs non-MBR biological (2015-2019)", d15[d15.f_physchem == 0].reset_index(drop=True), "f_mbr", 1.0,
                   base_nums + ["cod_eff", "tn_in", "tn_eff"], binary=True, drop_flags=("f_mbr", "f_two_stage"))
results.append(r3)
r4, yr4, tr4 = run("Scale (per 10x volume)", d, "log_vol", 1.0, ["cod_in", "cod_eff", "year"])
results.append(r4)
r5, _, _ = run("Influent COD (per +100 mg/L)", d, "cod_in", 100.0, ["log_vol", "cod_eff", "year"])
results.append(r5)
pd.DataFrame(results).to_csv(OUT / "dml_ate.csv", index=False)

# ---- heterogeneity: group ATEs (GATE) of the cleaner-COD effect by scale class, period, technology
print("\n=== GATE: cleaner-COD effect by subgroup (same residuals, subgroup-wise PLR) ===")
q = d["q_m3d"]
d["scale_class"] = pd.cut(q, [0, 1e3, 1e4, 5e4, 1e5, 1e9], labels=["<1k", "1k-10k", "10k-50k", "50k-100k", ">100k"])
d["period"] = pd.cut(d["year"], [2005, 2010, 2015, 2019], labels=["2006-10", "2011-15", "2016-19"])
d["eff_class"] = pd.cut(d["cod_eff"], [0, 20, 30, 50, 2000], labels=["<=20", "20-30", "30-50", ">50"])
gate = []
for col in ["scale_class", "period", "tech", "eff_class", "region"]:
    for lv, ix in d.groupby(col, observed=True).indices.items():
        if len(ix) < 400:
            continue
        r = plr(yr1[ix], tr1[ix], d["group_id"].values[ix])
        gate.append(dict(by=col, level=str(lv), n=len(ix), **{k: r[k] for k in ["theta", "se", "lo", "hi"]}))
gate = pd.DataFrame(gate)
gate["pct"] = (10 ** gate.theta - 1) * 100
gate.to_csv(OUT / "dml_gate_cod.csv", index=False)
print(gate.round(4).to_string(index=False))

# ---- dose-response: binned residual-on-residual (nonlinearity of the stringency effect)
print("\n=== dose-response: mean residual log ei by effluent-COD bin (after partialling out confounders) ===")
bins = [0, 15, 20, 25, 30, 40, 50, 60, 2000]
d["bin"] = pd.cut(d["cod_eff"], bins)
dr = pd.DataFrame({"bin": d["bin"], "yr": yr1}).groupby("bin", observed=True)["yr"].agg(["mean", "sem", "count"])
dr["pct_vs_mean"] = (10 ** dr["mean"] - 1) * 100
dr.to_csv(OUT / "dose_response_cod.csv")
print(dr.round(4).to_string())

# ---- within-plant first differences on reconstructed trajectories (plant fixed effects)
print("\n=== within-plant first differences (reconstructed trajectories) ===")
L = pd.read_parquet(ROOT / "results" / "02c_linkage" / "links.parquet")
full = pd.read_parquet(ROOT / "results" / "cache" / "model_table.parquet").sort_values("no").reset_index(drop=True)
a, b = full.loc[L.i.values].reset_index(drop=True), full.loc[L.j.values].reset_index(drop=True)
fd = pd.DataFrame({
    "dy": b.log_ei.values - a.log_ei.values, "dT": -(b.cod_eff.values - a.cod_eff.values) / 10.0,
    "dlv": b.log_vol.values - a.log_vol.values, "dci": (b.cod_in.values - a.cod_in.values) / 100.0,
    "year": b.year.values, "how": L.how.values, "province": b.province.values, "pid": b.plant_id.values,
    "ok": (a.complete.values & b.complete.values)})
fd = fd[fd.ok & np.isfinite(fd.dT)]
import statsmodels.formula.api as smf
fdres = []
for label, sub in [("all strict links", fd), ("changed records only (|dT|>0 & dvol!=0)", fd[(fd.dT != 0) & (fd.dlv != 0)]),
                   ("identity-neutral fills only", fd[fd.how == 3]),
                   ("fills, changed only", fd[(fd.how == 3) & (fd.dT != 0) & (fd.dlv != 0)])]:
    if len(sub) < 200:
        continue
    mdl = smf.ols("dy ~ dT + dlv + dci + C(year)", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["pid"]})
    ci = mdl.conf_int().loc["dT"]
    fdres.append(dict(sample=label, n=len(sub), theta=mdl.params["dT"], lo=ci[0], hi=ci[1], scale_elast=mdl.params["dlv"],
                      influent=mdl.params["dci"]))
    print(f"  {label:42s} n={len(sub):6d}  cleaner-COD per 10 mg/L: {mdl.params['dT']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"({(10**mdl.params['dT']-1)*100:+.1f}%)  dlogV: {mdl.params['dlv']:+.3f}  dCODin/100: {mdl.params['dci']:+.4f}")
pd.DataFrame(fdres).to_csv(OUT / "first_difference.csv", index=False)
np.savez_compressed(OUT / "residuals_cod.npz", yr=yr1, tr=tr1)
print("\nsaved to", OUT)
