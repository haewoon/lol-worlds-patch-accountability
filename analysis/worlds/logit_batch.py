"""Unpenalized logit MLEs for many exposure permutations of the same outcomes.

Each fit has exactly the w8 design: intercept, pre-event rating logit, exposure
difference. Batched Newton steps change computation, not the statistic.
"""
import numpy as np
from scipy.special import expit


def exposure_coefficients(outcome, rating, exposures, tol=1e-9, maxiter=40, return_studentized=False):
    y, x = np.asarray(outcome, float), np.asarray(rating, float)
    d = np.atleast_2d(np.asarray(exposures, float))
    if d.shape[1] != len(y) or x.shape != y.shape:
        raise ValueError("Inconsistent game dimensions")
    theta = np.zeros((len(d), 3))
    theta[:, 0] = np.log(y.mean() / (1 - y.mean()))
    h = np.empty((len(d), 3, 3))
    for _ in range(maxiter):
        p = expit(theta[:, 0, None] + theta[:, 1, None] * x + theta[:, 2, None] * d)
        w, residual = p * (1 - p), y - p
        score = np.column_stack([residual.sum(axis=1), residual @ x, (residual * d).sum(axis=1)])
        h[:, 0, 0] = w.sum(axis=1)
        h[:, 0, 1] = h[:, 1, 0] = w @ x
        h[:, 0, 2] = h[:, 2, 0] = (w * d).sum(axis=1)
        h[:, 1, 1] = w @ (x * x)
        h[:, 1, 2] = h[:, 2, 1] = (w * d) @ x
        h[:, 2, 2] = (w * d * d).sum(axis=1)
        step = np.linalg.solve(h, score[..., None])[..., 0]
        theta += step
        if np.max(np.abs(step)) < tol:
            if return_studentized:
                unit = np.zeros((len(d), 3, 1))
                unit[:, 2, 0] = 1.0
                variance = np.linalg.solve(h, unit)[:, 2, 0]
                return theta[:, 2], theta[:, 2] / np.sqrt(variance)
            return theta[:, 2]
    raise RuntimeError("Batched logistic MLE did not converge")
