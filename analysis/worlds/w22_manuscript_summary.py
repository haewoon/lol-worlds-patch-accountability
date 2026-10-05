"""Historical descriptive summaries; never rewrites the frozen 2026 forecast."""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from worlds_common import OUT

D = pd.read_csv(OUT / "buff_persistence_teams.csv")
validation, by_year = [], []
for dep in ["intended_z", "buffs_z", "nerfs_z"]:
    correlations = []
    for year in sorted(D.year.unique()):
        fit = smf.ols(f"{dep} ~ breadth_z + meta_z", D[D.year != year]).fit()
        held = D[D.year == year]
        correlations.append(spearmanr(fit.predict(held), held[dep]).statistic)
        by_year.append({"exposure": dep, "year": year, "spearman": correlations[-1]})
    validation.append({"exposure": dep, "loyo_spearman_mean": np.mean(correlations),
                       "loyo_spearman_min": np.min(correlations), "loyo_spearman_max": np.max(correlations),
                       "loyo_spearman_n_positive": int(np.sum(np.array(correlations) > 0)), "loyo_years": len(correlations)})
pd.DataFrame(validation).to_csv(OUT / "historical_style_validation.csv", index=False)
pd.DataFrame(by_year).to_csv(OUT / "historical_style_validation_by_year.csv", index=False)
E = pd.read_csv(OUT / "team_patch_exposure.csv").dropna(subset=["intended"])
C = pd.read_csv(OUT / "buff_persistence_champion_presence.csv")
C["decile"] = np.minimum(np.ceil(C.pres_pct * 10).astype(int), 10)
S = {"eligible_team_years": len(E), "champion_years": len(C),
     "unseen_min": int(E.n_unseen_patches.min()), "unseen_max": int(E.n_unseen_patches.max()),
     "unseen_mean": E.n_unseen_patches.mean()}
for c in ["buff_share", "nerf_share"]:
    S[c + "_mean_pct"] = E[c].mean() * 100
    S[c + "_median_range_pp"] = E.groupby("year")[c].agg(lambda s: s.max()-s.min()).median() * 100
for name, frame in [("top_decile", C[C.decile == 10]), ("bottom_eight", C[C.decile <= 8])]:
    S[name + "_buff_pct"] = (frame.net > 0).mean() * 100
    S[name + "_nerf_pct"] = (frame.net < 0).mean() * 100
K = pd.read_csv(OUT / "knockout_patch_exposure.csv")
top = K.loc[K.groupby("year").buffs.idxmax()]
S["most_buff_exposed_finals"] = int(top.finish.isin(["Champion", "Runner-up"]).sum())
S["most_buff_exposed_titles"] = int(top.finish.eq("Champion").sum())
S["highest_net_titles"] = int(K.loc[K.groupby("year").intended.idxmax()].finish.eq("Champion").sum())
t1 = []
for year in range(2022, 2026):
    qf = D[(D.year == year) & D.finish.notna()].copy()
    qf["breadth_rank"] = qf.breadth.rank(ascending=False)
    qf["meta_low_rank"] = qf.meta.rank()
    r = qf[qf.team.eq("T1")].iloc[0]
    t1.append({"year": year, "breadth_rank": r.breadth_rank, "meta_low_rank": r.meta_low_rank})
S["T1_qf_style_ranks"] = t1
(OUT / "manuscript_descriptives.json").write_text(json.dumps(S, indent=2) + "\n")
print(json.dumps(S, indent=2))
print(pd.DataFrame(validation).to_string(index=False))
