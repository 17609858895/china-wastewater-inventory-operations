# -*- coding: utf-8 -*-
"""Bootstrap site-level methane factors by process and project them onto the inventory. The optional model comparison estimates total GHG intensity."""
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, SEED, load_table, design_matrix, splits, metrics

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "results" / "08_ch4_graybox"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
COMP = ["bio_ch4", "bio_n2o", "bio_co2", "elec_co2", "eff_ch4", "eff_n2o", "eff_co2"]
NJ = 4

# ---------------- (A) national totals
full = pd.read_parquet(ROOT / "results" / "cache" / "model_table.parquet")
tot = full.groupby("year")[COMP].sum() / 1e6          # Mt CO2e
tot["total"] = tot.sum(1)
tot.to_csv(OUT / "national_totals_Mt.csv")
print("national totals (Mt CO2e), selected years:\n", tot.loc[[2006, 2010, 2015, 2019]].round(2).to_string())
print("\n2019 shares:\n", (tot.loc[2019, COMP] / tot.loc[2019, "total"]).round(3).to_string())

# ---------------- (B) observation-constrained CH4
m = pd.read_parquet(ROOT / "results" / "cache" / "measured.parquet")
fam = {"AAO": "A2O", "AAO,AO": "A2O", "AAO,MBR": "MBR", "AO": "AO", "OD": "OD", "BAF": "Biofilm", "Other": "Other"}
m["fam"] = m["process"].map(fam)
site = m.groupby("site").agg(ef=("ef", "mean"), fam=("fam", "first"), city=("city", "first"), n=("ef", "size")).reset_index()
print(f"\nmeasured: {len(m)} records, {len(site)} sites; site-mean EF (kg CH4/kg COD removed): mean={site.ef.mean():.4f}, median={site.ef.median():.4f}")
rng = np.random.default_rng(SEED)
B = 4000
boot = {}
for f, g in list(site.groupby("fam")) + [("All", site)]:
    v = g.ef.values
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(B)])
    boot[f] = bs
meas = pd.DataFrame({f: dict(n_sites=int((site.fam == f).sum()) if f != "All" else len(site), mean=b.mean(),
                             lo=np.quantile(b, .025), hi=np.quantile(b, .975)) for f, b in boot.items()}).T
full["ef_inv"] = full["bio_ch4"] / 28.0 * 1000 / full["cod_removed_kg"]          # kg CH4 / kg COD removed
inv = full.groupby("tech")["ef_inv"].median()
meas["inventory_EF"] = [inv.get(f, np.nan) for f in meas.index]
meas["ratio_meas_to_inventory"] = meas["mean"] / meas["inventory_EF"]
meas.to_csv(OUT / "ch4_ef_measured_vs_inventory.csv")
print(meas.round(5).to_string())

AEROBIC = ["A2O", "AO", "OD", "SBR", "CAS", "MBR", "Biofilm", "Bio-unspecified"]
key = full["tech"].where(full["tech"].isin(["A2O", "AO", "OD", "MBR", "Biofilm"]), "All")
aer = full["tech"].isin(AEROBIC).values
years = np.arange(2006, 2020)
# Each draw uses a common factor for all records in its process class.
# Sum the inventory's COD load by year and class before scaling each draw.
load = full.loc[aer].groupby([key[aer], "year"]).cod_removed_kg.sum().unstack(fill_value=0)
load = load.reindex(columns=years, fill_value=0)
draws = np.column_stack([boot[family] for family in load.index])
unchanged = full.loc[~aer].groupby("year").bio_ch4.sum().reindex(years, fill_value=0)
scen = (draws @ load.to_numpy() * 28 / 1000 + unchanged.to_numpy()) / 1e6
sc = pd.DataFrame({"year": years, "inventory_bio_CH4_Mt": tot["bio_ch4"].values, "obs_constrained_Mt": scen.mean(0),
                   "lo": np.quantile(scen, .025, 0), "hi": np.quantile(scen, .975, 0)})
sc["ratio"] = sc.obs_constrained_Mt / sc.inventory_bio_CH4_Mt
sc["obs_Gg_CH4"] = sc.obs_constrained_Mt / 28 * 1000
sc.to_csv(OUT / "ch4_observation_constrained_by_year.csv", index=False)
print("\nobservation-constrained on-site CH4 (Mt CO2e):\n", sc[sc.year.isin([2007, 2015, 2019])].round(3).to_string(index=False))
t19 = tot.loc[2019].copy()
new_total = t19["total"] - t19["bio_ch4"] + sc.loc[sc.year == 2019, "obs_constrained_Mt"].iloc[0]
print(f"2019 total: inventory {t19['total']:.2f} Mt -> obs-constrained {new_total:.2f} Mt; on-site CH4 share "
      f"{t19['bio_ch4']/t19['total']:.1%} -> {sc.loc[sc.year==2019,'obs_constrained_Mt'].iloc[0]/new_total:.1%}")

if "--factors-only" in sys.argv:
    sys.exit(0)

# ---------------- (C) gray-box (identity-anchored) vs black-box, no electricity as input
d = load_table()
d["total"] = d[COMP].sum(1)
vol_m3 = d["vol_1e4m3"].values * 1e4
y = np.log10(d["total"].values * 1000 / vol_m3)                                   # log10 kg CO2e / m3
Xdf, nums = design_matrix(d)
X = Xdf.values.astype(np.float32)
Xg = np.column_stack([X, d["grid_ef"].values.astype(np.float32)])                 # black-box gets the grid factor too
Xc = np.column_stack([Xg, np.log10(d["ei"].values).astype(np.float32)])           # circular: electricity given
d["proc"] = d["proc_major"] + " | " + d["proc_sub1"] + " | " + d["proc_sub2"]
tn_in_kg = d["tn_in"].values * vol_m3 / 1e3
nonelec = d[["bio_ch4", "bio_n2o", "bio_co2", "eff_ch4", "eff_n2o", "eff_co2"]].sum(1).values
eff_part = d[["eff_ch4", "eff_n2o", "eff_co2"]].sum(1).values                     # exact functions of cod_eff, tn_eff, pathway


def lgb():
    return LGBMRegressor(n_estimators=300, learning_rate=0.06, num_leaves=31, min_child_samples=60, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, n_jobs=NJ, random_state=SEED, verbose=-1)


rows = []
for scheme in ["plant", "province", "temporal"]:
    for k, (tr, te) in enumerate(splits(d, scheme)):
        # black-box
        p_bb = lgb().fit(Xg[tr], y[tr]).predict(Xg[te])
        p_cc = lgb().fit(Xc[tr], y[tr]).predict(Xc[te])
        # gray-box: emission-factor lookup learnt on the training fold + ML for electricity intensity
        trd = d.iloc[tr]
        lk = pd.DataFrame({"proc": trd["proc"].values,
                           "ch4": trd["bio_ch4"].values / trd["cod_removed_kg"].values,
                           "co2": trd["bio_co2"].values / trd["cod_removed_kg"].values,
                           "n2o": trd["bio_n2o"].values / tn_in_kg[tr]}).groupby("proc").median()
        glob = lk.median()
        ted = d.iloc[te]
        f = lk.reindex(ted["proc"].values)
        for c in lk.columns:
            f[c] = f[c].fillna(glob[c])
        bio_hat = (f["ch4"].values + f["co2"].values) * ted["cod_removed_kg"].values + f["n2o"].values * tn_in_kg[te]
        ei_hat = 10 ** lgb().fit(X[tr], d["log_ei"].values[tr]).predict(X[te])
        elec_hat = ei_hat * vol_m3[te] * d["grid_ef"].values[te] / 1000.0          # t CO2
        p_gb = np.log10((bio_hat + eff_part[te] + elec_hat) * 1000 / vol_m3[te])
        for nm, p in [("Black-box (no electricity)", p_bb), ("Identity-anchored gray-box (no electricity)", p_gb),
                      ("Black-box given electricity (circular)", p_cc)]:
            r = dict(scheme=scheme, fold=k, model=nm); r.update(metrics(y[te], p)); rows.append(r)
        print(f"  {scheme} fold{k}: black {rows[-3]['R2']:.3f} | gray {rows[-2]['R2']:.3f} | circular {rows[-1]['R2']:.3f}", flush=True)
G = pd.DataFrame(rows)
G.to_csv(OUT / "graybox_long.csv", index=False)
S = G.groupby(["scheme", "model"])[["R2", "RMSE", "MAE", "MdAPE"]].mean().reset_index()
S.to_csv(OUT / "graybox_summary.csv", index=False)
print("\n", S.round(4).to_string(index=False))
print("\nshare of electricity CO2 in plant total (median):", round(float((d.elec_co2 / d.total).median()), 3))
