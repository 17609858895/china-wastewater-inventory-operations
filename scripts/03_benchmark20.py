# -*- coding: utf-8 -*-
"""03 | 20 种回归算法 × 多种划分方案，目标 = log10(吨水电耗 kWh/m3)。
用法: python 03_benchmark20.py [schemes逗号分隔] [n_workers]
输出: results/03_benchmark/metrics_long.csv, summary.csv, oof_*.parquet
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, SEED, TARGET, load_table, design_matrix, splits, metrics

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "03_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
NT = 4  # threads per model


def make_models(num_idx):
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.kernel_approximation import Nystroem
    from sklearn.svm import LinearSVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                                  GradientBoostingRegressor, HistGradientBoostingRegressor)
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    sc = lambda: StandardScaler()
    spl = ColumnTransformer([("s", SplineTransformer(n_knots=8, degree=3), num_idx)], remainder="passthrough")
    poly = ColumnTransformer([("p", make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False)), num_idx)],
                             remainder="passthrough")
    return {
        "Scale-only": "scale_only",
        "OLS": make_pipeline(sc(), LinearRegression()),
        "Ridge": make_pipeline(sc(), Ridge(alpha=10.0)),
        "Lasso": make_pipeline(sc(), Lasso(alpha=5e-4, max_iter=5000)),
        "ElasticNet": make_pipeline(sc(), ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000)),
        "BayesRidge": make_pipeline(sc(), BayesianRidge()),
        "Huber": make_pipeline(sc(), HuberRegressor(alpha=1e-3, max_iter=400)),
        "PolyRidge": make_pipeline(poly, Ridge(alpha=5.0)),
        "SplineGAM": make_pipeline(spl, Ridge(alpha=1.0)),
        "KNN": make_pipeline(sc(), KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=NT)),
        "SVR": make_pipeline(sc(), Nystroem(kernel="rbf", gamma=0.01, n_components=600, random_state=SEED),
                             LinearSVR(epsilon=0.02, C=1.0, max_iter=3000, random_state=SEED)),
        "MLP": make_pipeline(sc(), MLPRegressor((128, 64), alpha=1e-3, early_stopping=True, max_iter=300,
                                                random_state=SEED)),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=30, random_state=SEED),
        "RandomForest": RandomForestRegressor(300, min_samples_leaf=3, max_features=0.4, n_jobs=NT, random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(300, min_samples_leaf=3, max_features=0.5, n_jobs=NT, random_state=SEED),
        "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=80, learning_rate=0.5,
                                      random_state=SEED),
        "GBDT": GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                                          random_state=SEED),
        "HistGB": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                                min_samples_leaf=40, l2_regularization=1.0, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=NT, random_state=SEED),
        "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=NT,
                                  random_state=SEED, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=1000, learning_rate=0.08, depth=8, thread_count=NT, verbose=0,
                                      random_seed=SEED),
    }


def run_one(name, scheme, k, tr, te, X, y, num_idx):
    t0 = time.time()
    model = make_models(num_idx)[name]
    Xtr, Xte, ytr = X[tr], X[te], y[tr]
    if isinstance(model, str):      # scale-only power law: log ei ~ a + b log vol
        b = np.polyfit(Xtr[:, 0], ytr, 1)
        ptr, pte = np.polyval(b, Xtr[:, 0]), np.polyval(b, Xte[:, 0])
    else:
        model.fit(Xtr, ytr)
        ptr, pte = model.predict(Xtr), model.predict(Xte)
    row = dict(model=name, scheme=scheme, fold=k, n_train=len(tr), n_test=len(te), secs=time.time() - t0)
    row.update({f"test_{a}": b for a, b in metrics(y[te], pte).items()})
    row.update({f"train_{a}": b for a, b in metrics(ytr, ptr).items()})
    print(f"  {scheme:9s} {name:13s} fold{k}  test R2={row['test_R2']:.3f}  train R2={row['train_R2']:.3f}  {row['secs']:.0f}s", flush=True)
    return row, (name, scheme, k, te, pte)


if __name__ == "__main__":
    schemes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["random", "province", "temporal"]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    d = load_table()
    Xdf, nums = design_matrix(d)
    X = Xdf.values.astype(np.float32)
    y = d[TARGET].values.astype(np.float64)
    num_idx = list(range(len(nums)))
    print("n =", len(d), "p =", X.shape[1], "schemes =", schemes)
    names = list(make_models(num_idx).keys())
    jobs = []
    for s in schemes:
        if s == "plant" and "plant_id" not in d:
            continue
        for k, (tr, te) in enumerate(splits(d, s)):
            for nm in names:
                jobs.append((nm, s, k, tr, te))
    res = Parallel(n_jobs=workers, verbose=0)(delayed(run_one)(nm, s, k, tr, te, X, y, num_idx) for nm, s, k, tr, te in jobs)
    rows = pd.DataFrame([r for r, _ in res])
    tag = "_".join(schemes)
    rows.to_csv(OUT / f"metrics_long_{tag}.csv", index=False)
    oof = {}
    for _, (nm, s, k, te, p) in res:
        oof.setdefault(s, pd.DataFrame({"row": np.arange(len(d)), "y": y}).set_index("row"))
        if nm not in oof[s]:
            oof[s][nm] = np.nan
        oof[s].loc[te, nm] = p
    for s, o in oof.items():
        o.to_parquet(OUT / f"oof_{s}.parquet")
    summ = (rows.groupby(["scheme", "model"])
            .agg(test_R2=("test_R2", "mean"), test_R2_min=("test_R2", "min"), test_RMSE=("test_RMSE", "mean"),
                 test_MAE=("test_MAE", "mean"), test_MdAPE=("test_MdAPE", "mean"), train_R2=("train_R2", "mean"),
                 secs=("secs", "sum")).reset_index())
    summ["gap"] = summ["train_R2"] - summ["test_R2"]
    summ.to_csv(OUT / f"summary_{tag}.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    for s in schemes:
        print(f"\n===== {s} =====")
        print(summ[summ.scheme == s].sort_values("test_R2", ascending=False).round(4).to_string(index=False))
