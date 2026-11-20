# -*- coding: utf-8 -*-
"""Reconstruct strict longitudinal links and broader modeling groups from the inventory."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
OUT = ROOT / "results" / "02c_linkage"
OUT.mkdir(parents=True, exist_ok=True)
MAX_DLOGV = 0.35                       # 水量相差 >2.2 倍不允许匹配
TO, TF, MG = 4.4, 5.0, 1.0             # 锚点门槛（保序 / 无序 / 与次优间隔）
T2, KMAX = 1.0, 12                     # 填充门槛、片段最大长度
T_LOOSE = 2.4                          # 宽松分组门槛


def raw_score(a, b):
    dv = np.abs(a["lv"][:, None] - b["lv"][None, :])
    de = np.abs(a["le"][:, None] - b["le"][None, :])
    s = 2.0 * np.exp(-dv / 0.04) + 0.8 * np.exp(-de / 0.08)
    for key, w, scale in [("ce", 1.0, 0.08), ("ci", 0.6, 0.10)]:
        x, y = a[key][:, None], b[key][None, :]
        ok = np.isfinite(x) & np.isfinite(y)
        rel = np.abs(np.log10(np.where(ok, x, 1.0)) - np.log10(np.where(ok, y, 1.0)))
        s += np.where(ok, w * np.exp(-rel / scale), 0.25 * w)
    s += 0.4 * (a["pw"][:, None] == b["pw"][None, :])
    s += 0.3 * (a["tech"][:, None] == b["tech"][None, :])
    # 完全相同的上报数值（跨年照抄）几乎不可能偶然出现，给额外证据权重
    s += 1.2 * (dv < 1e-6) + 1.0 * (de < 1e-6)
    ci_a, ci_b = a["ci"][:, None], b["ci"][None, :]
    s += 0.8 * (np.isfinite(ci_a) & np.isfinite(ci_b) & (np.abs(ci_a - ci_b) < 1e-6))
    ce_a, ce_b = a["ce"][:, None], b["ce"][None, :]
    nonint = np.abs(ce_a - np.round(ce_a)) > 1e-3
    s += 0.6 * (np.isfinite(ce_a) & np.isfinite(ce_b) & (np.abs(ce_a - ce_b) < 1e-6) & nonint)
    s[dv > MAX_DLOGV] = -1e9
    return s


def neutral_score(a, b):
    """身份中性得分：不使用电耗（结果变量）与出水浓度（处理变量）。
    证据 = 水量接近 + 进水 COD 接近 + 去向/工艺/设施类型一致 + 水量或进水 COD 完全相同。"""
    dv = np.abs(a["lv"][:, None] - b["lv"][None, :])
    s = 2.0 * np.exp(-dv / 0.08) + 0.4 * (a["pw"][:, None] == b["pw"][None, :]) + 1.2 * (dv < 1e-6)
    x, y = a["ci"][:, None], b["ci"][None, :]
    ok = np.isfinite(x) & np.isfinite(y)
    rel = np.abs(np.log10(np.where(ok, x, 1.0)) - np.log10(np.where(ok, y, 1.0)))
    s += np.where(ok, 0.8 * np.exp(-rel / 0.10) + 0.8 * (rel < 1e-9), 0.2)
    s += 0.3 * (a["tech"][:, None] == b["tech"][None, :]) + 0.3 * (a["fac"][:, None] == b["fac"][None, :])
    s[dv > MAX_DLOGV] = -1e9
    return s


@njit(cache=True)
def nw_align(s):
    n, m = s.shape
    H = np.zeros((n + 1, m + 1))
    T = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = H[i - 1, j]; t = 1
            if H[i, j - 1] > best:
                best = H[i, j - 1]; t = 2
            d = H[i - 1, j - 1] + s[i - 1, j - 1]
            if s[i - 1, j - 1] > 0 and d > best:
                best = d; t = 3
            H[i, j] = best; T[i, j] = t
    ia = np.full(n, -1, dtype=np.int64)
    i, j = n, m
    while i > 0 and j > 0:
        if T[i, j] == 3:
            ia[i - 1] = j - 1; i -= 1; j -= 1
        elif T[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return ia


def anchors(R, theta_o=TO, theta_f=TF, margin=MG):
    ia = nw_align(R - theta_o)
    how = np.where(ia >= 0, 1, 0)
    if theta_f is not None:
        freeA = np.where(ia < 0)[0]
        usedB = np.zeros(R.shape[1], bool); usedB[ia[ia >= 0]] = True
        freeB = np.where(~usedB)[0]
        if len(freeA) and len(freeB):
            S = R[np.ix_(freeA, freeB)]
            bj = S.argmax(1); bs = S.max(1); bi = S.argmax(0)
            S2 = S.copy(); S2[np.arange(len(freeA)), bj] = -1e9
            second_row = S2.max(1)
            S3 = S.copy(); S3[bi, np.arange(len(freeB))] = -1e9
            second_col = S3.max(0)
            for k in range(len(freeA)):
                j = bj[k]
                if bs[k] >= theta_f and bi[j] == k and bs[k] - max(second_row[k], second_col[j]) >= margin:
                    ia[freeA[k]] = freeB[j]; how[freeA[k]] = 2
    return ia, how


def segments(ia):
    """相邻保序锚点之间的片段 [(a_lo, a_hi, b_lo, b_hi)]（开区间内的记录）。只用单调的锚点链。"""
    idx = np.where(ia >= 0)[0]
    if len(idx) == 0:
        return []
    # 取最长单调子序列，保证片段有意义
    js = ia[idx]
    keep, last = [], -1
    for i, j in zip(idx, js):
        if j > last:
            keep.append((i, j)); last = j
    return [(i1 + 1, i2, j1 + 1, j2) for (i1, j1), (i2, j2) in zip(keep[:-1], keep[1:])]


def fill(ia, how, N, segs, shift=0):
    """在锚间片段内用中性得分补链。shift>0 时把 B 片段换成后面第 shift 个片段（安慰剂）。"""
    n_fill = 0
    out = []
    for k, (a0, a1, b0, b1) in enumerate(segs):
        if shift:
            if k + shift >= len(segs):
                continue
            _, _, b0, b1 = segs[k + shift]
        la, lb = a1 - a0, b1 - b0
        if la < 1 or lb < 1 or la > KMAX or lb > KMAX:
            continue
        S = N[a0:a1, b0:b1]
        th = T2 - (0.6 if la == lb else 0.0)          # 片段等长时放宽
        loc = nw_align(S - th)
        for i in np.where(loc >= 0)[0]:
            out.append((a0 + i, b0 + loc[i])); n_fill += 1
    if not shift:
        for i, j in out:
            if ia[i] < 0 and not (ia == j).any():
                ia[i] = j; how[i] = 3
    return n_fill


def pack(g):
    return dict(lv=g["log_vol"].values, le=np.log10(g["elec_1e4kwh"].values), ce=g["cod_eff"].values,
                ci=g["cod_in"].values, pw=g["pathway"].values, tech=g["tech"].values, fac=g["facility"].values)


d = pd.read_parquet(CACHE / "model_table.parquet").sort_values("no").reset_index(drop=True)
OLD_GROUP = d.set_index("no")["group_id"] if "group_id" in d else None
d = d.drop(columns=[c for c in ["plant_id", "group_id", "link_score_prev", "link_how_prev", "first_year", "n_years"] if c in d])
provs = list(d["province"].unique())
order = list(d.groupby("province").size().sort_values().index)
neighbour = {p: order[(order.index(p) + 1) % len(order)] for p in order}
groups = {k: g for k, g in d.groupby(["province", "year"], sort=False)}

def year_pairs():
    for prov in provs:
        for y in range(2006, 2019):
            A, B, C = groups.get((prov, y)), groups.get((prov, y + 1)), groups.get((neighbour[prov], y + 1))
            if A is None or B is None:
                continue
            pa, pb = pack(A), pack(B)
            yield dict(prov=prov, y=y, A=A.index.values, B=B.index.values, R=raw_score(pa, pb),
                              N=neutral_score(pa, pb), Rc=raw_score(pa, pack(C)) if C is not None else None)

nA = sum(len(g) for (province, year), g in groups.items() if (province, year + 1) in groups)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("final", "scan", "fillscan"), nargs="?", default="final")
parser.add_argument("secondary_threshold", type=float, nargs="?", default=2.8)
parser.add_argument("segment_limit", type=int, nargs="?", default=12)
parser.add_argument("preserve", choices=("keepgroup",), nargs="?")
parser.add_argument("--groups-from", type=Path, help="Archived model_table.parquet with the historical no/group_id mapping.")
args = parser.parse_args()
mode = args.mode
if mode == "scan":
    rows = []
    for to in [2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8]:
        real = sum(int((nw_align(p["R"] - to) >= 0).sum()) for p in year_pairs())
        plc = sum(int((nw_align(p["Rc"] - to) >= 0).sum()) for p in year_pairs() if p["Rc"] is not None)
        rows.append(dict(theta=to, linked=real, rate=real / nA, placebo=plc, fp_rate=plc / nA, fdr_est=plc / real))
    r = pd.DataFrame(rows); r.to_csv(OUT / "threshold_scan.csv", index=False)
    print(r.round(4).to_string(index=False)); sys.exit()

if mode == "fillscan":
    rows = []
    for t2 in [1.0, 1.6, 2.0, 2.4, 2.8, 3.2]:
        for km in [8, 12]:
            T2, KMAX = t2, km
            nf = npl = 0
            for p in year_pairs():
                ia, how = anchors(p["R"])
                segs = segments(np.where(how == 1, ia, -1))
                npl += fill(ia.copy(), how.copy(), p["N"], segs, shift=1)
                fill(ia, how, p["N"], segs)
                nf += int((how == 3).sum())
            rows.append(dict(T2=t2, KMAX=km, fills=nf, placebo=npl, fdr_est=npl / max(nf, 1)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "fill_threshold_scan.csv", index=False); sys.exit()
T2, KMAX = args.secondary_threshold, args.segment_limit
KEEP_GROUP = args.preserve == "keepgroup" or args.groups_from is not None
if args.groups_from:
    group_table = pd.read_parquet(args.groups_from, columns=["no", "group_id"])
    if group_table["no"].duplicated().any() or group_table.group_id.isna().any():
        raise ValueError("The archived mapping must have one nonmissing group for each record number.")
    OLD_GROUP = group_table.set_index("no")["group_id"]
    if len(OLD_GROUP) != len(d) or not d["no"].isin(OLD_GROUP.index).all():
        raise ValueError("Record numbers in the archived mapping and the inventory differ.")
if KEEP_GROUP and OLD_GROUP is None:
    raise ValueError("No historical groups found. Supply --groups-from or run a fresh grouping without keepgroup.")

links, rep = [], []
tot = dict(anchor=0, anchor_placebo=0, fill=0, fill_placebo=0, fill_eligible_A=0, loose=0)
loose_links = []
for p in year_pairs():
    ia, how = anchors(p["R"])
    n_anchor = int((ia >= 0).sum())
    segs = segments(np.where(how == 1, ia, -1))
    n_fill_plc = fill(ia.copy(), how.copy(), p["N"], segs, shift=1)
    elig = sum(a1 - a0 for a0, a1, b0, b1 in segs if 1 <= a1 - a0 <= KMAX and 1 <= b1 - b0 <= KMAX)
    fill(ia, how, p["N"], segs)
    n_fill = int((how == 3).sum())
    n_plc = int((anchors(p["Rc"])[0] >= 0).sum()) if p["Rc"] is not None else 0
    for i in np.where(ia >= 0)[0]:
        links.append((p["A"][i], p["B"][ia[i]], p["R"][i, ia[i]], how[i]))
    il = nw_align(p["R"] - T_LOOSE)
    for i in np.where(il >= 0)[0]:
        loose_links.append((p["A"][i], p["B"][il[i]]))
    tot["anchor"] += n_anchor; tot["anchor_placebo"] += n_plc; tot["fill"] += n_fill
    tot["fill_placebo"] += n_fill_plc; tot["fill_eligible_A"] += elig; tot["loose"] += int((il >= 0).sum())
    rep.append((p["prov"], p["y"], len(p["A"]), len(p["B"]), n_anchor, n_fill, n_plc, n_fill_plc, elig))

rep = pd.DataFrame(rep, columns=["province", "year", "nA", "nB", "anchors", "fills", "anchor_placebo", "fill_placebo", "fill_eligible"])
rep.to_csv(OUT / "linkage_report_province_year.csv", index=False)
print(f"records in years 2006-2018: {nA}")
print(f"stage-1 anchors: {tot['anchor']} ({tot['anchor']/nA:.1%}); cross-province placebo: {tot['anchor_placebo']} "
      f"-> est. false-link share {tot['anchor_placebo']/tot['anchor']:.1%}")
print(f"stage-2 fills  : {tot['fill']} of {tot['fill_eligible_A']} eligible in-segment records; shifted-segment placebo: "
      f"{tot['fill_placebo']} -> est. false-link share {tot['fill_placebo']/max(tot['fill'],1):.1%}")
print(f"total strict links: {tot['anchor']+tot['fill']} ({(tot['anchor']+tot['fill'])/nA:.1%});  loose links (CV grouping only): {tot['loose']} ({tot['loose']/nA:.1%})")
print("\nby year:")
print(rep.groupby("year")[["nA", "anchors", "fills", "anchor_placebo", "fill_placebo"]].sum()
      .assign(rate=lambda x: (x.anchors + x.fills) / x.nA).round(3).to_string())

L = pd.DataFrame(links, columns=["i", "j", "score", "how"])


def chain(n, pairs_ij):
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in pairs_ij:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return pd.factorize(np.array([find(i) for i in range(n)]))[0]


d["plant_id"] = chain(len(d), zip(L.i.values, L.j.values))
d["group_id"] = chain(len(d), list(zip(L.i.values, L.j.values)) + loose_links)
if KEEP_GROUP and OLD_GROUP is not None:      # 交叉验证分组保持与此前所有结果一致
    d["group_id"] = d["no"].map(OLD_GROUP).values
d["link_how_prev"] = 0
d.loc[L.j.values, "link_how_prev"] = L.how.values
d["link_score_prev"] = np.nan
d.loc[L.j.values, "link_score_prev"] = L.score.values
spell = d.groupby("plant_id")["year"].agg(["min", "max", "count"])
d["first_year"] = d["plant_id"].map(spell["min"]); d["n_years"] = d["plant_id"].map(spell["count"])
print(f"\nstrict trajectories: {d.plant_id.nunique()} pseudo-plants; loose CV groups: {d.group_id.nunique()}  (records {len(d)})")
print("strict trajectory length distribution:\n", spell["count"].value_counts().sort_index().to_string())
print("records in strict trajectories >=2 yrs:", int((d.n_years >= 2).sum()), " >=3:", int((d.n_years >= 3).sum()), " >=5:", int((d.n_years >= 5).sum()))
print("max records per plant-year (strict / loose):", d.groupby(["plant_id", "year"]).size().max(), d.groupby(["group_id", "year"]).size().max())

a, b = d.loc[L.i.values], d.loc[L.j.values]
def cont_table(a, b):
    return pd.DataFrame({
        "abs_dlog_vol": np.abs(a.log_vol.values - b.log_vol.values),
        "abs_dlog_ei": np.abs(a.log_ei.values - b.log_ei.values),
        "abs_dlog_cod_in": np.abs(np.log10(a.cod_in.values) - np.log10(b.cod_in.values)),
        "abs_dlog_cod_eff": np.abs(np.log10(a.cod_eff.values) - np.log10(b.cod_eff.values)),
        "same_pathway": (a.pathway.values == b.pathway.values).astype(float),
        "same_tech": (a.tech.values == b.tech.values).astype(float)})
cont = cont_table(a, b)
rb = b.groupby(["province", "year"], group_keys=False).sample(frac=1, random_state=1)
cont_r = cont_table(a, rb)
tab = pd.DataFrame({"anchor_median": cont[L.how.values < 3].median(), "fill_median": cont[L.how.values == 3].median(),
                    "anchor_mean": cont[L.how.values < 3].mean(), "fill_mean": cont[L.how.values == 3].mean(),
                    "random_median": cont_r.median(), "random_mean": cont_r.mean()})
print("\nP3 continuity: anchors vs identity-neutral fills vs random same-province-year pairs")
print(tab.round(3).to_string())
tab.to_csv(OUT / "continuity_table.csv")
print("\ntech label changed between consecutive linked years, by year:")
print(pd.DataFrame({"year": a.year.values, "changed": 1 - cont.same_tech.values}).groupby("year")["changed"].mean().round(3).to_string())
L.to_parquet(OUT / "links.parquet", index=False)
d.to_parquet(CACHE / "model_table.parquet", index=False)
pd.Series(tot).to_json(OUT / "linkage_totals.json")
print("\nsaved plant_id / group_id into model_table.parquet")
