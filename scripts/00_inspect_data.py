# -*- coding: utf-8 -*-
"""00 | 只读核查原始工作簿，并缓存为 parquet（不改写原表）。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
import argparse
parser = argparse.ArgumentParser(description="Read the two published workbooks and build local parquet caches.")
parser.add_argument("inventory", type=Path, help="Inventory workbook from Figshare")
parser.add_argument("methane", type=Path, help="Atmospheric methane workbook from Zenodo")
args = parser.parse_args()
XLSX, MEAS = args.inventory, args.methane
CACHE = ROOT / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)

xl = pd.ExcelFile(XLSX)
print("SHEETS:", xl.sheet_names)

firm = pd.read_excel(XLSX, sheet_name="Firm level GHG emissions")
firm = firm.loc[:, [c for c in firm.columns if not str(c).startswith("Unnamed")]]
firm.columns = ["no", "year", "province", "facility", "proc_major", "proc_sub1", "proc_sub2",
                "pathway", "elec_1e4kwh", "vol_1e4m3", "cod_removed_kg", "tn_removed_kg",
                "bio_ch4", "bio_n2o", "bio_co2", "elec_co2", "eff_ch4", "eff_n2o", "eff_co2"]
print("\nSHAPE:", firm.shape)
print(firm.head(8).to_string())
print("\nDTYPES:\n", firm.dtypes)
print("\nMISSING:\n", firm.isna().sum())
num = firm.select_dtypes("number").columns.drop(["no"])
print("\nDESCRIBE:\n", firm[num].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.to_string())
print("\nZERO / NEG COUNTS:")
for c in num:
    print(f"  {c:16s} zero={int((firm[c]==0).sum()):6d} neg={int((firm[c]<0).sum()):5d} nan={int(firm[c].isna().sum()):5d}")
for c in ["province", "facility", "proc_major", "proc_sub1", "proc_sub2", "pathway"]:
    vc = firm[c].value_counts(dropna=False)
    print(f"\n== {c}: {vc.shape[0]} levels ==")
    print(vc.head(40).to_string())

firm.to_parquet(CACHE / "firm.parquet", index=False)

for sh in xl.sheet_names:
    if sh == "Firm level GHG emissions":
        continue
    d = pd.read_excel(XLSX, sheet_name=sh, header=None)
    print(f"\n######## {sh}  shape={d.shape}")
    print(d.head(30).to_string())

m = pd.read_excel(MEAS)
print("\n######## MEASURED", m.shape)
print(m.head(20).to_string())
m.columns = ["site", "city", "date", "ef", "process"]
m["site"] = m["site"].ffill()
print(m.describe(include="all").T.to_string())
print(m.groupby("city")["ef"].describe().to_string())
print(m.groupby("process")["ef"].describe().to_string())
print("n sites:", m["site"].nunique())
m.to_parquet(CACHE / "measured.parquet", index=False)
