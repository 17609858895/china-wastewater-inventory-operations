"""Summarize facility contributions to the 2019 electricity gaps and linked-pair persistence."""
import numpy as np
import pandas as pd

from common import ROOT


def rank_and_persistence(results):
    frontier = pd.read_parquet(results / "06_frontier/frontier_oof.parquet")
    table = pd.read_parquet(results / "cache/model_table.parquet").sort_values("no").reset_index(drop=True)
    links = pd.read_parquet(results / "02c_linkage/links.parquet")
    year = frontier.loc[frontier.year == 2019]
    gap = year.save_q25.sort_values(ascending=False)
    print(f"2019 electricity (TWh): {year.elec_1e4kwh.sum() / 1e5:.3f}")
    for quantile in ("q50", "q25", "q10"):
        print(f"2019 {quantile} positive reference gap (TWh): {year['save_' + quantile].sum() / 1e9:.3f}")
    print(f"Top tenth of facilities' share of q25 gap: {gap.iloc[:len(gap) // 10].sum() / gap.sum():.2%}")
    print(f"Facilities above 10,000 m3/day, share of q10 gap: {year.loc[year.q_m3d > 10000, 'save_q10'].sum() / year.save_q10.sum():.2%}")

    residuals = (frontier.log_ei - frontier.q50).set_axis(frontier.no)
    a, b = table.iloc[links.i], table.iloc[links.j]
    ra, rb = residuals.reindex(a.no).to_numpy(), residuals.reindex(b.no).to_numpy()
    changed = np.abs(a.log_ei.to_numpy() - b.log_ei.to_numpy()) > 1e-9
    available = np.isfinite(ra) & np.isfinite(rb) & changed
    for label, kind in (("All", links.how > 0), ("Anchors", links.how < 3), ("Identity-oriented", links.how == 3)):
        keep = available & kind.to_numpy()
        print(f"{label}: {keep.sum()} changed-intensity pairs, correlation {np.corrcoef(ra[keep], rb[keep])[0, 1]:.4f}")


if __name__ == "__main__":
    rank_and_persistence(ROOT / "results")
