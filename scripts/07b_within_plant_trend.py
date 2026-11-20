# -*- coding: utf-8 -*-
"""Compare changes in pooled geometric mean electricity intensity against chained changes in linked pairs."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
full = pd.read_parquet(ROOT / "results" / "cache" / "model_table.parquet").sort_values("no").reset_index(drop=True)
L = pd.read_parquet(ROOT / "results" / "02c_linkage" / "links.parquet")
a, b = full.loc[L.i.values].reset_index(drop=True), full.loc[L.j.values].reset_index(drop=True)
fd = pd.DataFrame({"year": b.year.values, "dy": b.log_ei.values - a.log_ei.values, "how": L.how.values,
                   "dvol": b.log_vol.values - a.log_vol.values, "dce": b.cod_eff.values - a.cod_eff.values})
fd["changed"] = fd.dy.abs() > 1e-9
allm = full.groupby("year").log_ei.mean()
rows = []
for y, g in fd.groupby("year"):
    ch = g[g.changed]
    nf = g[g.how == 3]
    rows.append(dict(year=y, n_links=len(g), within_all=g.dy.mean(), within_changed=ch.dy.mean(), se_changed=ch.dy.std() / np.sqrt(len(ch)),
                     within_neutral=nf.dy.mean() if len(nf) > 30 else np.nan, n_neutral=len(nf),
                     pooled_change=allm.loc[y] - allm.loc[y - 1]))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.round(4).to_string(index=False))
post = r[r.year >= 2016]
print("\ncumulative 2015->2019 (sum of yearly changes, log10): pooled cross-section = %.4f (%+.1f%%); within-plant all links = %.4f (%+.1f%%); "
      "within-plant changed records = %.4f (%+.1f%%); identity-neutral links = %.4f (%+.1f%%)" % (
          post.pooled_change.sum(), (10 ** post.pooled_change.sum() - 1) * 100,
          post.within_all.sum(), (10 ** post.within_all.sum() - 1) * 100,
          post.within_changed.sum(), (10 ** post.within_changed.sum() - 1) * 100,
          post.within_neutral.sum(), (10 ** post.within_neutral.sum() - 1) * 100))
# new entrants vs incumbents in 2016-2019: records without a link to the previous year
full["has_prev"] = False
full.loc[L.j.values, "has_prev"] = True
t = full[full.year >= 2016].groupby(["year", "has_prev"]).agg(n=("ei", "size"), median_ei=("ei", "median"), median_q=("q_m3d", "median")).round(3)
print("\nlinked (incumbent) vs unlinked records, 2016-2019:\n", t.to_string())
(ROOT / "results" / "07_change_attribution").mkdir(parents=True, exist_ok=True)
r.to_csv(ROOT / "results" / "07_change_attribution" / "within_plant_trend.csv", index=False)
