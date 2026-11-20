# -*- coding: utf-8 -*-
"""Invert the published inventory equations and construct the plant-year feature table."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300)

d = pd.read_parquet(CACHE / "firm.parquet")
vol_m3 = d["vol_1e4m3"] * 1e4

RIVER = ["Direct discharge into rivers, lakes and reserviors etc",
         "Enter into sewers first, then discharge into rivers, lakes and reserviors"]
SEA = ["Direct discharge into seas", "Enter into sewers first, then discharge into seas"]
K_CH4 = pd.Series(np.nan, index=d.index)
K_CH4[d.pathway.isin(RIVER)] = 0.25 * 0.19 * 28 / 1000 * 10      # 0.0133
K_CH4[d.pathway.isin(SEA)] = 0.25 * 0.035 * 28 / 1000 * 10       # 0.00245
K_N2O = 0.005 * 1.58 * 265 / 1000 * 10                            # 0.020935

d["cod_eff"] = d["eff_ch4"] / d["vol_1e4m3"] / K_CH4              # mg/L
d["tn_eff"] = np.where(d["eff_n2o"] > 0, d["eff_n2o"] / d["vol_1e4m3"] / K_N2O, np.nan)
d["cod_rem"] = d["cod_removed_kg"] / vol_m3 * 1e3                 # mg/L
d["tn_rem"] = d["tn_removed_kg"] / vol_m3 * 1e3
d["cod_in"] = d["cod_rem"] + d["cod_eff"]
d["tn_in"] = d["tn_rem"] + d["tn_eff"]
d["cod_re"] = d["cod_rem"] / d["cod_in"]                          # removal efficiency
d["tn_re"] = d["tn_rem"] / d["tn_in"]
d["cn_in"] = d["cod_in"] / d["tn_in"]
d["ei"] = d["elec_1e4kwh"] / d["vol_1e4m3"]                       # kWh/m3
d["ei_cod"] = d["elec_1e4kwh"] * 1e4 / d["cod_removed_kg"]        # kWh/kg COD removed
d["grid_ef"] = d["elec_co2"] / d["elec_1e4kwh"] / 10              # kg CO2/kWh
d["log_vol"] = np.log10(vol_m3)
d["log_ei"] = np.log10(d["ei"])
d["q_m3d"] = vol_m3 / 365.0

# --- lattice check: share of recovered effluent values that are (near-)integers
for c in ["cod_eff", "tn_eff"]:
    v = d[c].dropna()
    near_int = (np.abs(v - v.round()) < 1e-3).mean()
    near_1dp = (np.abs(v * 10 - (v * 10).round()) < 1e-3).mean()
    near_2dp = (np.abs(v * 100 - (v * 100).round()) < 1e-3).mean()
    print(f"{c}: n={len(v)}, integer share={near_int:.3f}, <=1 decimal share={near_1dp:.3f}, <=2 decimals share={near_2dp:.3f}")
    print(v.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
    print("  top values:", v.round(3).value_counts().head(12).to_dict())

# --- does bio N2O follow influent TN (not removed TN)?
d["proc"] = d["proc_major"] + " | " + d["proc_sub1"] + " | " + d["proc_sub2"]
t = d[(d.bio_n2o > 0) & d.tn_in.notna()].copy()
t["r_rem"] = t.bio_n2o / t.tn_removed_kg
t["r_in"] = t.bio_n2o / (t.tn_in * t.vol_1e4m3 * 10)
for col in ["r_rem", "r_in"]:
    g = t.groupby("proc")[col]
    cv = (g.std() / g.mean()).dropna()
    print(f"bio_N2O ratio {col}: median within-process CV = {cv.median():.4g}; "
          f"share of processes with CV<1e-6 = {(cv < 1e-6).mean():.2%}")
top = t["proc"].value_counts().head(6).index
print(t[t.proc.isin(top)].groupby("proc")["r_in"].agg(["min", "median", "max"]).to_string())

# --- process flags (engineering-readable)
txt = (d["proc_major"] + ";" + d["proc_sub1"] + ";" + d["proc_sub2"]).str.lower()
d["f_a2o"] = txt.str.contains("a2o").astype(int)
d["f_ao"] = txt.str.contains(r"anoxic/oxic \(ao\)").astype(int)
d["f_od"] = txt.str.contains("oxidation ditch").astype(int)
d["f_sbr"] = txt.str.contains("sbr").astype(int)
d["f_mbr"] = txt.str.contains("membrane bioreactor").astype(int)
d["f_biofilm"] = txt.str.contains("biofilm|biofilter|contact oxidation|rotating biological").astype(int)
d["f_anaer"] = txt.str.contains("anaerobic biological|anaerobic biofilter|anaerobic hydrolysis|anaerobic lagoon|typical anaerobic").astype(int)
d["f_natural"] = txt.str.contains("wetland|lagoon|land infiltration|stabilization").astype(int)
d["f_physchem"] = d["proc_major"].isin(["Physical Treatment", "Chemical Treatment", "Physicochemical Treatment",
                                        "Physical and Chemical Treatment"]).astype(int)
d["f_two_stage"] = (d["proc_sub2"] != "-").astype(int)
d["tech"] = np.select(
    [d.f_mbr == 1, d.f_a2o == 1, d.f_od == 1, d.f_sbr == 1, d.f_ao == 1, d.f_biofilm == 1, d.f_anaer == 1,
     d.f_natural == 1, d.f_physchem == 1, d.proc_major == "Activated Sludge"],
    ["MBR", "A2O", "OD", "SBR", "AO", "Biofilm", "Anaerobic", "Natural", "PhysChem", "CAS"], default="Bio-unspecified")
print("\ntech family counts:\n", d["tech"].value_counts().to_string())

REGION = {
    "North": ["Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia"],
    "Northeast": ["Liaoning", "Jilin", "Heilongjiang"],
    "East": ["Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong"],
    "Central": ["Henan", "Hubei", "Hunan"],
    "South": ["Guangdong", "Guangxi", "Hainan"],
    "Southwest": ["Chongqing", "Sichuan", "Guizhou", "Yunnan", "Tibet"],
    "Northwest": ["Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"],
}
p2r = {p: r for r, ps in REGION.items() for p in ps}
d["region"] = d["province"].map(p2r)
assert d["region"].notna().all()

d["complete"] = d[["cod_eff", "tn_eff"]].notna().all(axis=1) & (d["cod_eff"] > 0)
print("\ncomplete-case rows (river/sea pathways with both effluent values):", int(d.complete.sum()), f"({d.complete.mean():.1%})")
m = d[d.complete]
print(m[["ei", "ei_cod", "q_m3d", "cod_in", "cod_eff", "cod_re", "tn_in", "tn_eff", "tn_re", "cn_in", "grid_ef"]]
      .describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.round(3).to_string())
print("\nby year medians (complete cases):")
print(m.groupby("year")[["ei", "cod_in", "cod_eff", "tn_in", "tn_eff", "cod_re", "tn_re", "f_mbr", "f_a2o", "q_m3d"]]
      .agg({"ei": "median", "cod_in": "median", "cod_eff": "median", "tn_in": "median", "tn_eff": "median",
            "cod_re": "median", "tn_re": "median", "f_mbr": "mean", "f_a2o": "mean", "q_m3d": "median"}).round(3).to_string())
print("\nshare meeting Class 1A-like effluent (COD<=50 & TN<=15) by year:")
print(m.assign(c1a=(m.cod_eff <= 50) & (m.tn_eff <= 15)).groupby("year")["c1a"].mean().round(3).to_string())
print("\nelectricity intensity by tech family (complete cases):")
print(m.groupby("tech")["ei"].describe(percentiles=[.1, .5, .9]).round(3).to_string())
print("\nSpearman correlations with log ei:")
print(m[["log_ei", "log_vol", "cod_in", "cod_eff", "tn_in", "tn_eff", "cod_re", "tn_re", "cn_in", "year"]]
      .corr(method="spearman")["log_ei"].round(3).to_string())

d.to_parquet(CACHE / "model_table.parquet", index=False)
print("\nsaved:", CACHE / "model_table.parquet", d.shape)
