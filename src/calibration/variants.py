"""Model variants used in calibration (step 03) and cross-validation (step 03b)."""
from src.models import gipps, idm, newell, ovrv

MODELS = {"idm": idm, "ovrv": ovrv, "newell": newell, "gipps": gipps}

# name: (model, parameter spec key in experiment.yaml, run selection)
VARIANTS = {
    "main_idm":          ("idm",    "idm",      "main"),
    "main_ovrv":         ("ovrv",   "ovrv",     "main"),
    "supp_idm_free":     ("idm",    "idm_free", "main"),
    "supp_idm_allruns":  ("idm",    "idm",      "supplementary"),
    "supp_ovrv_allruns": ("ovrv",   "ovrv",     "supplementary"),
    # EV-aware extensions (actuation lag) and baselines with explicit reaction time
    "ext_idm_lag":       ("idm",    "idm_lag",  "main"),
    "ext_ovrv_lag":      ("ovrv",   "ovrv_lag", "main"),
    "base_newell":       ("newell", "newell",   "main"),
    "base_gipps":        ("gipps",  "gipps",    "main"),
}
