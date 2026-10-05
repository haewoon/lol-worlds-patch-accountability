"""W16: model-conditional power using the same studentized statistic as w8.

Outcomes are independent Bernoulli draws on the fixed observed schedule; ratings,
exposures and their measurement are held fixed. This is not a causal validity test.
The 95% pointwise intervals describe Monte Carlo error across 500 simulated outcomes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from worlds_common import OUT
from logit_batch import exposure_coefficients

rng = np.random.default_rng(16)
N_SIM, N_PERM, ALPHA = 500, 1000, 0.05
GRID = np.array([0, .06, .10, .12, .14, .16, .20, .24,
                 -.06, -.10, -.12, -.14, -.16, -.20, -.24])
W = pd.read_parquet(OUT / "worlds_games_pred.parquet").sort_values("date").reset_index(drop=True)
E = pd.read_csv(OUT / "team_patch_exposure.csv").dropna(subset=["intended_z"])
ez = E.set_index(["year", "teamid"]).intended_z
W["d"] = ez.reindex(list(zip(W.year, W.blue_id))).to_numpy() - ez.reindex(list(zip(W.year, W.red_id))).to_numpy()
D = W.dropna(subset=["d"]).reset_index(drop=True)
Xb = sm.add_constant(D[["pred_logit"]]).to_numpy()
m0 = sm.Logit(D.result, Xb).fit(disp=0)
base = Xb @ m0.params
DP = np.empty((N_PERM, len(D)))
for k in range(N_PERM):
    col = pd.Series(np.nan, index=D.index)
    for yy, group in E.groupby("year"):
        mp = dict(zip(group.teamid, group.intended_z.to_numpy()[rng.permutation(len(group))]))
        use = D.year == yy
        col[use] = [mp[b] - mp[r] for b, r in zip(D.blue_id[use], D.red_id[use])]
    DP[k] = col.to_numpy()
assert np.isfinite(DP).all()
d = D.d.to_numpy()
all_d = np.vstack([d, DP])
x = D.pred_logit.to_numpy()
# Independent implementation check against the actual statsmodels fits.
b, z = exposure_coefficients(D.result, x, all_d[:26], return_studentized=True)
ref = [sm.Logit(D.result, np.column_stack([np.ones(len(D)), x, q])).fit(disp=0)
       for q in all_d[:26]]
err = float(np.max(np.abs(b - [m.params.iloc[2] for m in ref])))
zerr = float(np.max(np.abs(z - [m.tvalues.iloc[2] for m in ref])))
assert err < 1e-8 and zerr < 1e-8
(OUT / "power_protocol.json").write_text(json.dumps({
    "statistic": "absolute exposure coefficient / model-based standard error from refitted unpenalized logit",
    "n_simulations_per_effect": N_SIM, "permutations": N_PERM,
    "alpha": ALPHA, "games": len(D), "seed": 16,
    "statsmodels_max_coefficient_error_26_fits": err,
    "statsmodels_max_z_error_26_fits": zerr,
    "assumptions": "independent Bernoulli outcomes; fixed schedule, ratings, exposure; no rating/coding uncertainty"
}, indent=2) + "\n")
curve = []
for effect in GRID:
    hits = 0
    for _ in range(N_SIM):
        y = (rng.random(len(D)) < expit(base + effect * d)).astype(float)
        _, statistics = exposure_coefficients(y, x, all_d, return_studentized=True)
        p = (np.sum(np.abs(statistics[1:]) >= abs(statistics[0])) + 1) / (N_PERM + 1)
        hits += p < ALPHA
    power = hits / N_SIM
    z = 1.96
    centre = (power + z*z/(2*N_SIM)) / (1+z*z/N_SIM)
    radius = z * np.sqrt(power*(1-power)/N_SIM + z*z/(4*N_SIM*N_SIM)) / (1+z*z/N_SIM)
    curve.append({"true_logit_per_sd": effect, "winprob_pp_per_sd": effect/4*100,
                  "power": power, "mc_lo": centre-radius, "mc_hi": centre+radius,
                  "rejections": hits, "n_sim": N_SIM})
    pd.DataFrame(curve).to_csv(OUT / "detectable_effect_power.csv", index=False)
    print(curve[-1], flush=True)
C = pd.DataFrame(curve)

def mde(sign):
    branch = C[C.true_logit_per_sd * sign >= 0].copy()
    branch["magnitude"] = branch.winprob_pp_per_sd.abs()
    branch = branch.sort_values("magnitude")
    if branch.power.max() < .8:
        raise ValueError("Power grid does not bracket 80%")
    return float(np.interp(.8, np.maximum.accumulate(branch.power), branch.magnitude))

def bo5(pp):
    q = .5 + np.asarray(pp)/100
    return (q**3 * (1+3*(1-q)+6*(1-q)**2)-.5)*100

pm = pd.read_csv(OUT / "patch_models.csv").set_index(["model", "term"]).loc[("intended", "d_intended_z")]
est, lo, hi = pm.estimate/4*100, pm.boot_lo/4*100, pm.boot_hi/4*100
pos, neg = mde(1), mde(-1)
summary = pd.DataFrame([
    {"scope": "game", "est": est, "lo": lo, "hi": hi, "mde80": pos, "mde80_negative": neg, "p_perm": pm.p_perm},
    {"scope": "bo5", "est": float(bo5(est)), "lo": float(bo5(lo)), "hi": float(bo5(hi)),
     "mde80": float(bo5(pos)), "mde80_negative": float(-bo5(-neg)), "p_perm": pm.p_perm}
])
summary.to_csv(OUT / "detectable_effect_summary.csv", index=False)
print(summary.to_string(index=False))
