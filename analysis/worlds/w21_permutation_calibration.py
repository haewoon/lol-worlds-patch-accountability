"""Paired null calibration before adopting a studentized permutation statistic.

Uses the same corrected sample, fixed within-year team permutations and simulated
outcomes for both statistics. The first 500 replicates reproduce w16's raw test.
This conditional model check does not establish exchangeability in observational data.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from worlds_common import OUT
from logit_batch import exposure_coefficients

N_SIM, N_PERM = 2000, 1000
rng = np.random.default_rng(16)
W = pd.read_parquet(OUT / "worlds_games_pred.parquet").sort_values("date").reset_index(drop=True)
E = pd.read_csv(OUT / "team_patch_exposure.csv").dropna(subset=["intended_z"])
ez = E.set_index(["year", "teamid"]).intended_z
W["d"] = ez.reindex(list(zip(W.year, W.blue_id))).to_numpy() - ez.reindex(list(zip(W.year, W.red_id))).to_numpy()
D = W.dropna(subset=["d"]).reset_index(drop=True)
x, y = D.pred_logit.to_numpy(), D.result.to_numpy()
base_X = sm.add_constant(x)
p0 = sm.Logit(y, base_X).fit(disp=0).predict(base_X)
DP = np.empty((N_PERM, len(D)))
for k in range(N_PERM):
    for yy, group in E.groupby("year"):
        mp = dict(zip(group.teamid, group.intended_z.to_numpy()[rng.permutation(len(group))]))
        ix = np.flatnonzero(D.year.to_numpy() == yy)
        DP[k, ix] = [mp[b] - mp[r] for b, r in zip(D.blue_id.iloc[ix], D.red_id.iloc[ix])]
all_d = np.vstack([D.d.to_numpy(), DP])
b, z = exposure_coefficients(y, x, all_d[:26], return_studentized=True)
reference = [sm.Logit(y, np.column_stack([np.ones(len(D)), x, d])).fit(disp=0) for d in all_d[:26]]
error_beta = float(np.max(np.abs(b - [m.params[2] for m in reference])))
error_z = float(np.max(np.abs(z - [m.tvalues[2] for m in reference])))
assert error_beta < 1e-8 and error_z < 1e-8
draws = []
for sim in range(N_SIM):
    ys = (rng.random(len(D)) < p0).astype(float)
    b, z = exposure_coefficients(ys, x, all_d, return_studentized=True)
    p_raw = (1 + np.sum(np.abs(b[1:]) >= abs(b[0]))) / (N_PERM + 1)
    p_student = (1 + np.sum(np.abs(z[1:]) >= abs(z[0]))) / (N_PERM + 1)
    draws.append({"simulation": sim + 1, "p_raw": p_raw, "p_studentized": p_student})
    if (sim + 1) % 250 == 0:
        current = pd.DataFrame(draws)
        current.to_csv(OUT / "permutation_null_calibration_draws.csv", index=False)
        print(sim + 1, ((current[["p_raw", "p_studentized"]] < .05).mean()).to_dict(), flush=True)
results = []
draws = pd.DataFrame(draws)
for n in [500, N_SIM]:
    for statistic in ["raw", "studentized"]:
        hits = int((draws.iloc[:n]["p_" + statistic] < .05).sum())
        rate, q = hits/n, 1.96
        centre = (rate + q*q/(2*n))/(1+q*q/n)
        radius = q*np.sqrt(rate*(1-rate)/n + q*q/(4*n*n))/(1+q*q/n)
        results.append({"statistic": statistic, "n_sim": n, "rejections": hits,
                        "rate": rate, "mc_lo": centre-radius, "mc_hi": centre+radius})
pd.DataFrame(results).to_csv(OUT / "permutation_null_calibration.csv", index=False)
(OUT / "permutation_calibration_protocol.json").write_text(json.dumps({
    "seed": 16, "n_perm": N_PERM, "n_sim": N_SIM,
    "statistic": "Wald z using inverse observed logit information; refitted for each permutation",
    "max_beta_error_vs_statsmodels": error_beta, "max_z_error_vs_statsmodels": error_z,
    "assumptions": "fixed observed schedule and covariates; independent Bernoulli outcomes from strength-only model"
}, indent=2) + "\n")
print(pd.DataFrame(results).to_string(index=False))
