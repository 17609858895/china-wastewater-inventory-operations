# -*- coding: utf-8 -*-
"""05 | (a) 堆叠集成（严格嵌套：元学习器只见训练折的内层 OOF）；
        (b) 按重建厂组划分的交叉验证（plant-grouped）；
        (c) 同一预测换算为"年用电量"的 R2；
        (d) 轨迹感知模型：加入上一年该厂的电耗强度（仅对有上一年链接的记录），并与"持续性基线"比较。
用法: python 05_stacking_plantcv.py
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, SEED, TARGET, load_table, design_matrix, splits, metrics

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "results" / "05_stacking"
OUT.mkdir(parents=True, exist_ok=True)
NT = 16


def base_models(num_idx):
    spl = ColumnTransformer([("s", SplineTransformer(n_knots=8, degree=3), num_idx)], remainder="passthrough")
    return {
        "LightGBM": lambda: LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=60,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0,
                                          n_jobs=NT, random_state=SEED, verbose=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=6, min_child_weight=10,
                                        subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, tree_method="hist",
                                        n_jobs=NT, random_state=SEED),
        "CatBoost": lambda: CatBoostRegressor(iterations=700, learning_rate=0.06, depth=7, l2_leaf_reg=5,
                                              thread_count=NT, verbose=0, random_seed=SEED),
        "ExtraTrees": lambda: ExtraTreesRegressor(250, min_samples_leaf=8, max_features=0.5, n_jobs=NT, random_state=SEED),
        "RandomForest": lambda: RandomForestRegressor(250, min_samples_leaf=8, max_features=0.4, n_jobs=NT, random_state=SEED),
        "SplineGAM": lambda: make_pipeline(spl, Ridge(alpha=1.0)),
    }


def inner_groups(d, scheme, tr):
    if scheme == "province":
        return d["province"].values[tr]
    return d["group_id"].values[tr]          # plant / temporal: never split a plant across inner folds


def run_scheme(d, X, y, num_idx, scheme):
    names = list(base_models(num_idx).keys())
    rows, oof = [], np.full((len(d), len(names) + 2), np.nan)
    wts = []
    for k, (tr, te) in enumerate(splits(d, scheme)):
        t0 = time.time()
        Z_tr = np.zeros((len(tr), len(names)))
        Z_te = np.zeros((len(te), len(names)))
        g_in = inner_groups(d, scheme, tr)
        n_in = min(4, len(np.unique(g_in)))
        for j, nm in enumerate(names):
            for a, b in GroupKFold(n_in).split(tr, groups=g_in):
                m = base_models(num_idx)[nm]().fit(X[tr[a]], y[tr[a]])
                Z_tr[b, j] = m.predict(X[tr[b]])
            Z_te[:, j] = base_models(num_idx)[nm]().fit(X[tr], y[tr]).predict(X[te])
        mu = y[tr].mean()
        w, _ = nnls(Z_tr - mu, y[tr] - mu)                    # non-negative stacking around the training mean
        p_stack = mu + (Z_te - mu) @ w
        p_avg = Z_te.mean(1)
        oof[te, :len(names)] = Z_te; oof[te, -2] = p_stack; oof[te, -1] = p_avg
        wts.append(dict(scheme=scheme, fold=k, **dict(zip(names, w))))
        for nm, p in list(zip(names, Z_te.T)) + [("Stacking(NNLS)", p_stack), ("Average", p_avg)]:
            r = dict(scheme=scheme, fold=k, model=nm, n_test=len(te))
            r.update(metrics(y[te], p))
            lv = d["log_vol"].values[te]
            yE, pE = y[te] + lv, p + lv                       # log10 annual electricity (up to a constant)
            r["R2_total_elec"] = 1 - ((yE - pE) ** 2).sum() / ((yE - yE.mean()) ** 2).sum()
            rows.append(r)
        print(f"  {scheme} fold{k} done in {time.time()-t0:.0f}s; weights={np.round(w,3)}", flush=True)
    cols = names + ["Stacking(NNLS)", "Average"]
    pd.DataFrame(oof, columns=cols).assign(y=y).to_parquet(OUT / f"oof_{scheme}.parquet")
    return pd.DataFrame(rows), pd.DataFrame(wts)


if __name__ == "__main__":
    d = load_table()
    Xdf, nums = design_matrix(d)
    X = Xdf.values.astype(np.float32); y = d[TARGET].values
    num_idx = list(range(len(nums)))
    allr, allw = [], []
    for s in ["plant", "province", "temporal"]:
        r, w = run_scheme(d, X, y, num_idx, s)
        allr.append(r); allw.append(w)
    R = pd.concat(allr); W = pd.concat(allw)
    R.to_csv(OUT / "stacking_long.csv", index=False); W.to_csv(OUT / "stacking_weights.csv", index=False)
    S = R.groupby(["scheme", "model"])[["R2", "RMSE", "MAE", "MdAPE", "R2_total_elec"]].mean().reset_index()
    S.to_csv(OUT / "stacking_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for s in ["plant", "province", "temporal"]:
        print(f"\n===== {s} =====")
        print(S[S.scheme == s].sort_values("R2", ascending=False).round(4).to_string(index=False))
    print("\nmean stacking weights:\n", W.groupby("scheme").mean(numeric_only=True).drop(columns="fold").round(3).to_string())

    # ---------- (d) trajectory-aware: previous-year intensity of the same (strictly linked) plant
    print("\n===== trajectory-aware model (records with a strict link to the previous year) =====")
    full = pd.read_parquet(ROOT / "results" / "cache" / "model_table.parquet").sort_values("no").reset_index(drop=True)
    L = pd.read_parquet(ROOT / "results" / "02c_linkage" / "links.parquet")
    prev = pd.Series(full.loc[L.i.values, "log_ei"].values, index=full.loc[L.j.values, "no"].values)
    d["lag_log_ei"] = d["no"].map(prev)
    sub = d[d.lag_log_ei.notna()].reset_index(drop=True)
    Xs, nums_s = design_matrix(sub)
    Xa = Xs.values.astype(np.float32)
    Xb = np.column_stack([Xa, sub["lag_log_ei"].values.astype(np.float32)])
    ys = sub[TARGET].values
    res = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(Xa, groups=sub["group_id"].values)):
        mk = lambda: LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=60,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, n_jobs=NT,
                                   random_state=SEED, verbose=-1)
        pa = mk().fit(Xa[tr], ys[tr]).predict(Xa[te])
        pb = mk().fit(Xb[tr], ys[tr]).predict(Xb[te])
        pp = sub["lag_log_ei"].values[te]                                    # persistence baseline
        chg = np.abs(ys[te] - pp) > 1e-9                                      # records that are not exact carry-overs
        for nm, p in [("Features only", pa), ("Features + plant history", pb), ("Persistence (last year)", pp)]:
            res.append(dict(model=nm, fold=k, R2=metrics(ys[te], p)["R2"], RMSE=metrics(ys[te], p)["RMSE"],
                            R2_changed=metrics(ys[te][chg], p[chg])["R2"], share_changed=chg.mean()))
    T = pd.DataFrame(res).groupby("model")[["R2", "RMSE", "R2_changed", "share_changed"]].mean()
    T.to_csv(OUT / "trajectory_aware.csv")
    print(f"n = {len(sub)}"); print(T.round(4).to_string())
