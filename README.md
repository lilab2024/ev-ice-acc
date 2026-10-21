# MixedTraffic-Modeling

Reproducible analysis for the revised paper on EV and ICE adaptive cruise control (ACC):
harmonized preprocessing of two non-harmonized ACC datasets, identifiability-aware
calibration, cross-validation with baselines, matched steady-state and event-level
comparison, a counterfactual same-leader test, string stability of homogeneous and mixed
platoons, and ring-road simulations.

## Quick start

**Only look at the results:** tables and figures of the paper are in `results/paper/`
(CSV and LaTeX tables, PDF and PNG figures). No code needs to be run.

**Reproduce everything** (about 4 hours on Google Colab; Python 3.13 with the package versions in `requirements.txt`):

    git clone https://github.com/lilab2024/ev-ice-acc.git
    cd ev-ice-acc
    pip install -r requirements.txt
    # place the data in data/ as described below (or set MTM_DATA_ROOT)
    python -m pytest tests -q        # check the installation
    bash run_all.sh                  # all steps in order

**Regenerate only the tables and figures** from the stored results (about 1 minute):

    python scripts/07_make_tables_figs.py

## Data

Not included in this repository. Download the original datasets and place them as

    data/raw/ice_acc/VehA_dip_min.csv, ...                 (Gunter et al., 2020)
    data/raw/ev_acc/<vehicle>/<gap>/<X>_desired/<v_free>/<v_reduced>/<run>.csv
                                                           (Lapardhaja et al., 2023)

By default the data are read from `data/` in the repository. To use another folder, set
`data_root` in `config/paths.yaml` (absolute, or relative to the repository root) or the
environment variable `MTM_DATA_ROOT`, e.g. in Google Colab:

    import os
    os.environ["MTM_DATA_ROOT"] = "/content/drive/MyDrive/<path to the data folder>"

Run identifiers keep the raw vehicle codes and folder names (e.g. `ICE_VehD_*` is VehC
in the paper, `EV_Tesla_*` is Model 3); see `config/vehicles.yaml`.

## Pipeline

Run from the repository root, in this order. Every step reads only the outputs of earlier
steps; long steps save each part immediately and can be resumed. Step 03 skips
configurations that already have a result; use `--force` to recompute them.
Measured run times on Google Colab: step 03 about 75 min, 03b about 100 min, 06 about
55 min, all other steps a few minutes or less.

| Step | Script | Output |
|---|---|---|
| 01 | `scripts/01_preprocess.py` | `data/processed/`: run manifest, QC table, 10 Hz trajectories |
| 03 | `scripts/03_calibrate.py --variants ...` | `results/calibration/` |
| 03b | `scripts/03b_cross_validate.py` | `results/cross_validation/` |
| 04 | `scripts/04_statistics.py` | `results/statistics/` |
| 05 | `scripts/05_stability.py` | `results/stability/` |
| 06 | `scripts/06_ring.py` | `results/ring/` |
| 07 | `scripts/07_make_tables_figs.py` | `results/paper/tables/`, `results/paper/figures/` |

Calibration variants: `main_idm main_ovrv supp_idm_free supp_idm_allruns
supp_ovrv_allruns ext_idm_lag ext_ovrv_lag base_newell base_gipps`.
All settings (bounds, seeds, run selection, simulation set-up) are in `config/experiment.yaml`.

## Tests

    python -m pytest tests -q

## Requirements

Python 3.13 with the package versions in `requirements.txt` (numpy, pandas, scipy,
matplotlib, pyyaml, pyarrow, pytest). All results were reproduced with exactly these
versions; with other versions the calibration may differ in the last digits.

## Paper tables and figures

Numbers are those of the manuscript; files are in `results/paper/tables/` and
`results/paper/figures/`. All are written by `scripts/07_make_tables_figs.py`, except
Table 1, a literature overview that is typeset by hand.

| Manuscript | File | Content |
|---|---|---|
| Table 2 | `table1_dataset` | Overview of the datasets |
| Table 3 | `tableS1_bounds` | Parameters and bounds of all models |
| Table 4 | `table2_steady_headway` | Steady-state time gap |
| Table 5 | `table3_events` | Response to leader decelerations |
| Table 6 | `table4_calibration` | Calibrated IDM parameters |
| Table 7 | `tableS3_identifiability` | IDM with all parameters free |
| Table 8 | `tableS4_allruns` | Calibration with and without stops |
| Table 9 | `table5_model_rmse` | Out-of-sample RMSE of all models |
| Table 10 | `tableS2_model_pairs` | Paired model comparisons |
| Table 11 | `table6_stability` | String stability |
| Fig. 1 | `fig1_framework` | Analysis framework |
| Fig. 2 | `figA_gap_definition` | Gap definitions (schematic) |
| Fig. 3 | `fig2a`-`fig2d` | Example runs |
| Fig. 4 | `fig3a`, `fig3b` | Steady-state time gap |
| Fig. 5 | `fig4a`, `fig4b` | Amplification and delay |
| Fig. 6 | `figBa_response_ice`, `figBb_response_ev` | Event-aligned speed response |
| Fig. 7 | `fig5a`, `fig5b` | Model fit |
| Fig. 8 | `fig6_validation` | Peak gain vs. measured amplification |
| Fig. 9 | `fig7_stability_margin` | Stability margin |
| Fig. 10 | `fig8a`, `fig8b` | Mixed platoons |
| Fig. 11 | `fig9a`-`fig9c` | Ring road, space-time diagrams |
| Fig. 12 | `fig10a`, `fig10b` | Ring road, growth rate |
| Fig. 13 | `figS1_flow_density` | Flow versus density |

File names keep the numbering of the code; LaTeX numbers the floats by their position.
Fig. 6 uses `results/data_summary/event_curves.csv`: it is rebuilt from the processed
data when they are available; otherwise the stored file is used.
