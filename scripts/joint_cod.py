"""Calculate electricity-intensity contrasts by daily flow and effluent COD (Fig. 3d)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from common import ROOT, SEED, design_matrix, load_table


def cross_fitted_residuals(data):
    """Fit the two nuisance regressions in the broader inventory linkage groups."""
    x = design_matrix(data, nums=["log_vol", "cod_in", "year"])[0].to_numpy(dtype=np.float32)
    responses = data[["log_ei", "cod_eff"]].to_numpy(copy=True)
    responses[:, 1] /= -10.0
    residuals = np.empty_like(responses)
    for train, test in GroupKFold(5).split(x, groups=data.group_id):
        for column in range(responses.shape[1]):
            model = LGBMRegressor(
                n_estimators=300, learning_rate=0.07, num_leaves=63,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4,
                random_state=SEED, verbose=-1,
            )
            model.fit(x[train], responses[train, column])
            residuals[test, column] = responses[test, column] - model.predict(x[test])
    return residuals.T


def summarize(data, outcome, exposure):
    flow = pd.cut(data.q_m3d, [0, 1000, 10000, 50000, 100000, 1e9])
    cod = pd.cut(data.cod_eff, [0, 20, 30, 50, 5000])
    rows = []
    for flow_bin in flow.cat.categories:
        for cod_bin in cod.cat.categories:
            mask = ((flow == flow_bin) & (cod == cod_bin)).to_numpy()
            groups = data.group_id.to_numpy()[mask]
            row = dict(flow_class=str(flow_bin), cod_class=str(cod_bin), records=int(mask.sum()),
                       linkage_groups=int(np.unique(groups).size))
            if mask.sum() >= 300:
                t, y = exposure[mask], outcome[mask]
                ss = t @ t
                theta = (t @ y) / ss
                scores = pd.Series(t * (y - t * theta)).groupby(groups).sum()
                se = np.linalg.norm(scores) / ss
                bounds = 100 * (10 ** (theta + np.array([0.0, -1.96, 1.96]) * se) - 1)
                row.update(zip(("contrast_pct", "lower_pct", "upper_pct"), bounds))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, help="Reuse an NPZ of aligned yr/tr residuals from the analysis archive.")
    args = parser.parse_args()
    data = load_table()
    if args.residuals:
        with np.load(args.residuals) as archive:
            y, t = archive["yr"], archive["tr"]
    else:
        y, t = cross_fitted_residuals(data)
    if len(y) != len(data) or len(t) != len(data):
        raise ValueError("The residual vectors must align with the complete records in inventory order.")
    output = ROOT / "results" / "joint_cod.csv"
    summarize(data, y, t).to_csv(output, index=False)
    print(output)
