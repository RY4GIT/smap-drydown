import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from MyLogger import getLogger

# Create a logger
log = getLogger(__name__)

__author__ = "Ryoko Araki"
__contact__ = "raraki@ucsb.edu"
__copyright__ = "Copyright 2024, SMAP-drydown project, @RY4GIT"
__license__ = "MIT"
__status__ = "Dev"
__url__ = ""


class Event:
    def __init__(self, index, event_dict):

        # Read the data
        self.index = index
        self.start_date = event_dict["event_start"]
        self.end_date = event_dict["event_end"]
        sm_subset = np.asarray(event_dict["sm_masked"])
        self.pet = np.nanmax(event_dict["PET"])
        self.min_sm = event_dict["min_sm"]
        self.max_sm = event_dict["max_sm"]
        self.est_theta_fc = event_dict["est_theta_fc"]
        self.est_theta_star = event_dict["est_theta_star"]

        # Prepare the attributes
        self.subset_sm_range = np.nanmax(sm_subset) - np.nanmin(sm_subset)
        self.subset_min_sm = np.nanmin(sm_subset)

        # Prepare the inputs
        t = np.arange(0, len(sm_subset), 1)
        self.x = t[~np.isnan(sm_subset)]
        self.y = sm_subset[~np.isnan(sm_subset)]

    def add_attributes(
        self,
        model_type="",
        popt=[],
        pcov=[],
        y_opt=[],
        r_squared=np.nan,
        aic=np.nan,
        aicc=np.nan,
        bic=np.nan,
        ss_res=np.nan,
        ss_tot=np.nan,
        p_value=np.nan,
        est_theta_star=np.nan,
        est_theta_w=np.nan,
    ):
        if model_type == "tau_exp":
            param_names = ["delta_theta", "theta_w", "tau"]
            var_delta_theta = pcov[
                param_names.index("delta_theta"), param_names.index("delta_theta")
            ]
            var_theta_w = pcov[
                param_names.index("theta_w"), param_names.index("theta_w")
            ]
            var_tau = pcov[param_names.index("tau"), param_names.index("tau")]
            cov_delta_theta_theta_w = pcov[
                param_names.index("delta_theta"), param_names.index("theta_w")
            ]
            cov_delta_theta_tau = pcov[
                param_names.index("delta_theta"), param_names.index("tau")
            ]
            cov_theta_w_tau = pcov[
                param_names.index("theta_w"), param_names.index("tau")
            ]

            self.tau_exp = {
                "delta_theta": popt[0],
                "theta_w": popt[1],
                "tau": popt[2],
                "var_delta_theta": var_delta_theta,
                "var_theta_w": var_theta_w,
                "var_tau": var_tau,
                "cov_delta_theta_theta_w": cov_delta_theta_theta_w,
                "cov_delta_theta_tau": cov_delta_theta_tau,
                "cov_theta_w_tau": cov_theta_w_tau,
                "y_opt": y_opt.tolist(),
                "r_squared": r_squared,
                "aic": aic,
                "aicc": aicc,
                "bic": bic,
                "ss_res": ss_res,
                "ss_tot": ss_tot
            }

        if model_type == "exp":
            param_names = ["ETmax", "theta_0", "theta_star"]

            var_ETmax = pcov[param_names.index("ETmax"), param_names.index("ETmax")]
            var_theta_0 = pcov[
                param_names.index("theta_0"), param_names.index("theta_0")
            ]
            if len(popt) == 3:
                var_theta_star = pcov[
                    param_names.index("theta_star"), param_names.index("theta_star")
                ]
            else:
                var_theta_star = np.nan

            # Extract covariances
            cov_ETmax_theta_0 = pcov[
                param_names.index("ETmax"), param_names.index("theta_0")
            ]

            if len(popt) == 3:
                cov_ETmax_theta_star = pcov[
                    param_names.index("ETmax"), param_names.index("theta_star")
                ]
                cov_theta_0_theta_star = pcov[
                    param_names.index("theta_0"), param_names.index("theta_star")
                ]
            else:
                cov_ETmax_theta_star = np.nan
                cov_theta_0_theta_star = np.nan

            self.exp = {
                "ETmax": popt[0],
                "theta_0": popt[1],
                "theta_star": est_theta_star,
                "theta_w": est_theta_w,
                "var_ETmax": var_ETmax,
                "var_theta_0": var_theta_0,
                "var_theta_star": var_theta_star,
                "cov_ETmax_theta_0": cov_ETmax_theta_0,
                "cov_ETmax_theta_star": cov_ETmax_theta_star,
                "cov_theta_0_theta_star": cov_theta_0_theta_star,
                "y_opt": y_opt.tolist(),
                "r_squared": r_squared,
                "aic": aic,
                "aicc": aicc,
                "bic": bic,
                "ss_res": ss_res,
                "ss_tot": ss_tot,
            }

        if model_type == "q":
            param_names = ["q", "ETmax", "theta_0", "theta_star"]

            # Extract variances
            var_q = pcov[param_names.index("q"), param_names.index("q")]
            var_ETmax = pcov[param_names.index("ETmax"), param_names.index("ETmax")]
            var_theta_0 = pcov[
                param_names.index("theta_0"), param_names.index("theta_0")
            ]
            if len(popt) == 4:
                var_theta_star = pcov[
                    param_names.index("theta_star"), param_names.index("theta_star")
                ]
            else:
                var_theta_star = np.nan

            # Extract covariances
            cov_q_ETmax = pcov[param_names.index("q"), param_names.index("ETmax")]
            cov_q_theta_0 = pcov[param_names.index("q"), param_names.index("theta_0")]
            cov_ETmax_theta_0 = pcov[
                param_names.index("ETmax"), param_names.index("theta_0")
            ]
            if len(popt) == 4:
                cov_q_theta_star = pcov[
                    param_names.index("q"), param_names.index("theta_star")
                ]
                cov_ETmax_theta_star = pcov[
                    param_names.index("ETmax"), param_names.index("theta_star")
                ]
                cov_theta_0_theta_star = pcov[
                    param_names.index("theta_0"), param_names.index("theta_star")
                ]
            else:
                cov_q_theta_star = np.nan
                cov_ETmax_theta_star = np.nan
                cov_theta_0_theta_star = np.nan

            self.q = {
                "q": popt[0],
                "ETmax": popt[1],
                "theta_0": popt[2],
                "theta_star": est_theta_star,
                "theta_w": est_theta_w,
                "var_q": var_q,
                "var_ETmax": var_ETmax,
                "var_theta_0": var_theta_0,
                "var_theta_star": var_theta_star,
                "cov_q_ETmax": cov_q_ETmax,
                "cov_q_theta_0": cov_q_theta_0,
                "cov_q_theta_star": cov_q_theta_star,
                "cov_ETmax_theta_0": cov_ETmax_theta_0,
                "cov_ETmax_theta_star": cov_ETmax_theta_star,
                "cov_theta_0_theta_star": cov_theta_0_theta_star,
                "y_opt": y_opt.tolist(),
                "r_squared": r_squared,
                "aic": aic,
                "aicc": aicc,
                "bic": bic,
                "ss_res": ss_res,
                "ss_tot": ss_tot,
                "q_eq_1_p": p_value,
            }

        if model_type == "sgm":
            self.sgm = {
                "theta50": popt[0],
                "k": popt[1],
                "a": popt[2],
                "r_squared": r_squared,
                "y_opt": y_opt.tolist(),
            }
