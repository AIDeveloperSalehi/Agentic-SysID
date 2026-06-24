"""
Uncertainty estimation for the fitted surrogate model.

GP:  posterior predictive std (already calibrated by GP noise hyperparameter).
NN:  heuristic — constant 10 % of output std (no dropout/laplace here).

Returns a dict of scalar summary statistics used by the report and dossier.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from agents.surrogate.model_class_selector import ModelClass
from agents.surrogate.trainer import SurrogatePredictor


class UncertaintyEstimator:
    """
    Computes predictive uncertainty statistics over the training set.
    """

    def estimate(
        self,
        model_class: ModelClass,
        predictor:   SurrogatePredictor,
        theta:       np.ndarray,
        theta_dot:   np.ndarray,
        u:           np.ndarray,
        theta_ddot:  np.ndarray,   # ground-truth (numerical derivative)
    ) -> Dict[str, float]:
        """
        Returns
        -------
        {
            "mean_std":    average predictive std over training points,
            "rmse":        RMSE on training set,
            "coverage_95": fraction of points where |error| < 1.96 * std,
        }
        """
        valid = (
            np.isfinite(theta)
            & np.isfinite(theta_dot)
            & np.isfinite(u)
            & np.isfinite(theta_ddot)
        )
        t_v  = theta[valid]
        td_v = theta_dot[valid]
        u_v  = u[valid]
        y_v  = theta_ddot[valid]

        if len(t_v) == 0:
            return {"mean_std": float("inf"), "rmse": float("inf"), "coverage_95": 0.0}

        mu, std = predictor.predict_with_std(t_v, td_v, u_v)
        errors  = mu - y_v
        rmse    = float(np.sqrt(np.mean(errors ** 2)))
        within  = np.abs(errors) < 1.96 * (std + 1e-12)
        cov95   = float(within.mean())

        return {
            "mean_std":    float(np.mean(std)),
            "rmse":        rmse,
            "coverage_95": cov95,
        }
