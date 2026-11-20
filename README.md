# Plant operations in China's wastewater emissions inventory

Python code for recovering plant operations from the 2006–2019 inventory, tracking changes in electricity intensity, estimating conditional electricity reference gaps, and propagating measured methane factors through the inventory.

## Inputs

Download both original Excel workbooks. The ingestion command accepts their paths in any directory.

| Workbook | Source | SHA-256 |
|---|---|---|
| `2006-2019 WWTP GHG emissions in China.xlsx` | [Wang et al., Figshare v1](https://doi.org/10.6084/m9.figshare.19158206.v1) | `ec653472a69966f0467d1adbce2209b8465f4cdb299b935db81a6ad5339d3d24` |
| `Measuring Wastewater treatment plants.xlsx` | [Sun et al., Zenodo v1](https://doi.org/10.5281/zenodo.18043683) | `036b9b178f191668a03489236401fdd74a36b29c4f724f31f634b51bf791af01` |

The methane workbook supplies site-level factors for the process classes. Its site codes are local to that workbook.

## Prepare the data and reconstruct plant links

Use Python 3.12 in a virtual environment. Run the following commands from the repository root, substituting the local paths to the two downloaded workbooks.

```sh
python -m pip install -r requirements.txt
python scripts/00_inspect_data.py "path/to/2006-2019 WWTP GHG emissions in China.xlsx" "path/to/Measuring Wastewater treatment plants.xlsx"
python scripts/02_build_dataset.py
python scripts/02c_link_plants.py final 2.8 12
python scripts/07b_within_plant_trend.py
```

The final secondary threshold is 2.8, reduced by 0.6 for equal-length segments. The segment length cap is 12. The source inventory contains 57,060 plant-years, with 52,152 complete records for electricity modeling. Strict linkage yields 13,950 anchors and 3,503 secondary pairs. Script 07b sums consecutive annual mean changes. The set of linked facilities can change between year pairs.

For models fitted with the historical broader groups, insert the following command before model fitting. Extract the original grouping table from the accompanying supporting analysis archive in a separate directory. The script reads only its inventory record number and broader group identifier, checks coverage, and recomputes the final strict links from the two original workbooks.

```sh
python scripts/02c_link_plants.py final 2.8 12 --groups-from "path/to/extracted/archive/results/cache/model_table.parquet"
```

Without an archive, the command above the grouping instructions builds fresh broader groups under the current linkage algorithm. Strict pairs, inventory reconstruction, chained intensity trends and the methane calculation are independent of that broader-group choice. The published broader grouping has 26,900 groups across all inventory records and predates final linkage refinement.

## Main analyses

| Run from the repository root | Output in `results/` |
|---|---|
| `python scripts/04_causal_dml.py` | Overall and marginal adjusted associations of effluent quality and plant characteristics with electricity intensity, grouped uncertainty, and linked first differences |
| `python scripts/joint_cod.py` | Joint flow-by-COD contrasts and grouped intervals in `joint_cod.csv`. Cells with fewer than 300 records have counts only |
| `python scripts/04b_mbr_total_direct.py` | Propensity overlap, adjusted MBR comparisons and estimates with effluent controls |
| `python scripts/06_frontier.py` | Cross-fitted electricity quantiles and annual positive gaps under three references |
| `python scripts/reference_summary.py` | Console summary of 2019 electricity gaps, ranked contributions and changed-pair residual correlations. Run after 06 |
| `python scripts/07_shapley_change.py` | Model-based decomposition of changes in mean log electricity intensity |
| `python scripts/08_ch4_graybox.py --factors-only` | Inventory totals and annual methane scenarios. This step requires only the two ingestion/reconstruction scripts, 00 and 02 |

For the methane scenario, each measurement site enters once after averaging repeat observations. Bootstrap draws are shared across all eligible inventory records within the relevant process class. The two samples have different observation periods. The reference electricity gaps are retrospective comparisons and represent candidates for follow-up metering. Intervention savings would require plant-level verification.

## Predictive benchmarks

```sh
python scripts/03_benchmark20.py random,province,temporal 1
python scripts/05_stacking_plantcv.py
python scripts/08_ch4_graybox.py
```

The final argument to 03 limits the number of simultaneous fitting jobs. The model benchmarks and full quantile fits can take substantially longer than reconstruction. Code identifiers such as `save_q25` and `total/direct` follow the original analysis files. The former is a positive electricity gap against a conditional quantile. The latter distinguishes adjustment sets, each with its own overlap sample.

## Interpreting a rerun

The dependency pins identify the reconstruction environment. Historical outer-fold membership vectors for the published predictive benchmarks were unavailable. Some saved predictions were fitted before final strict-link refinement. A fresh predictive run uses the supplied groups and installed estimators. Rounding, software versions and historical group assignments can affect agreement with archived results. The Supporting Information gives the reported benchmark summaries and describes the named-list comparisons. The archive accompanying the manuscript contains additional diagnostics. This repository hosts the principal numerical analysis scripts.
