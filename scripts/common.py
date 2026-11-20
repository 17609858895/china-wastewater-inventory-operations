# -*- coding: utf-8 -*-
"""公共工具：数据载入、特征矩阵、划分方案、指标。"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
SEED = 20261112

NUM = ["log_vol", "cod_in", "cod_eff", "cod_re", "tn_in", "tn_eff", "tn_re", "cn_in", "year"]
FLAGS = ["f_a2o", "f_ao", "f_od", "f_sbr", "f_mbr", "f_biofilm", "f_anaer", "f_natural", "f_physchem", "f_two_stage"]
CAT = ["province", "region", "facility", "proc_major", "proc_sub1", "proc_sub2", "pathway", "tech"]
TARGET = "log_ei"


def load_table(complete_only=True):
    d = pd.read_parquet(CACHE / "model_table.parquet")
    if complete_only:
        d = d[d["complete"]].reset_index(drop=True)
    return d


def design_matrix(d, min_count=30, cats=CAT, nums=NUM, flags=FLAGS):
    """数值 + 标志位 + 低频合并后的 one-hot。返回 (X DataFrame, 数值列名)。"""
    parts = [d[nums + flags].astype(float).reset_index(drop=True)]
    for c in cats:
        s = d[c].astype(str).reset_index(drop=True)
        vc = s.value_counts()
        keep = vc[vc >= min_count].index
        s = s.where(s.isin(keep), "_rare_")
        dm = pd.get_dummies(s, prefix=c, dtype=float)
        parts.append(dm)
    X = pd.concat(parts, axis=1)
    X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt").replace(",", ";") for c in X.columns]
    return X, list(nums)


def splits(d, scheme, n_splits=5, seed=SEED):
    """返回 [(train_idx, test_idx), ...]"""
    idx = np.arange(len(d))
    if scheme == "random":
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(idx))
    if scheme == "province":
        return list(GroupKFold(n_splits).split(idx, groups=d["province"].values))
    if scheme == "plant":          # 需要 02c 重建的伪厂 ID
        return list(GroupKFold(n_splits).split(idx, groups=d["plant_id"].values))
    if scheme == "temporal":
        return [(idx[d["year"].values <= 2016], idx[d["year"].values >= 2017])]
    if scheme == "rolling":
        out = []
        for cut, hi in [(2010, 2012), (2012, 2014), (2014, 2016), (2016, 2019)]:
            y = d["year"].values
            out.append((idx[y <= cut], idx[(y > cut) & (y <= hi)]))
        return out
    raise ValueError(scheme)


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=mean_absolute_error(y, p),
                MdAPE=float(np.median(np.abs(10 ** p / 10 ** y - 1))))
