#!/usr/bin/env bash
# Full reproduction of all results (about 7 hours). Run from the repository root.
set -e
python -m pytest tests -q
python scripts/01_preprocess.py
python scripts/03_calibrate.py --variants main_idm main_ovrv supp_idm_free \
    supp_idm_allruns supp_ovrv_allruns ext_idm_lag ext_ovrv_lag base_newell base_gipps
python scripts/03b_cross_validate.py
python scripts/04_statistics.py
python scripts/05_stability.py
python scripts/06_ring.py
python scripts/07_make_tables_figs.py
echo "Done. Tables and figures are in results/paper/"
