"""Check that the numbers in paper/main.tex match the pipeline outputs.

Each check recomputes or reads a number from the outputs, formats it as the manuscript does, and looks for that
text in main.tex; a few also refit a model independently. Run after the pipeline (run_revised_pipeline.py), the
validation scripts (w19 score, w25) and the coding sensitivity (w26). Set LOL_WORLDS_OUT to check another run.
"""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from worlds_common import OUT, ROOT, assign_series, load_matches, sr_only

class _MathInsensitive(str):
    """the manuscript sets numbers in math mode; text comparisons ignore the $ delimiters on both sides"""
    def __contains__(self, x):
        return str.__contains__(self, x.replace("$", ""))


TEX = _MathInsensitive((ROOT / "paper/main.tex").read_text().replace("$", ""))
results = []


def check(name, ok):
    results.append((name, bool(ok)))


def in_tex(name, text):
    check(f"{name}: {text!r}", text in TEX)


sgn = lambda x, d=1: f"${x:+.{d}f}$"          # e.g. $+0.8$, $-2.0$
pval = lambda p: f"$p = {p:.3f}$"


class _Words(dict):
    def __missing__(self, k):                    # numbers above ten are written as digits
        return str(k)


words = _Words(enumerate("zero one two three four five six seven eight nine ten".split()))

# ---------- sample ----------
G = pd.read_parquet(OUT / "worlds_games_pred.parquet")
E = pd.read_csv(OUT / "team_patch_exposure.csv")
PM = pd.read_csv(OUT / "patch_models.csv").set_index(["model", "term"])
coded_all = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in
                       sorted(glob.glob(str(ROOT / "data/patch_notes/coded_[A-E].csv")))])
coded = sr_only(coded_all)
NS = pd.read_csv(ROOT / "data/patch_notes/non_sr_rows.csv", dtype={"patch_oe": str})
check("Every listed non-Summoner's Rift row exists and is removed", len(coded_all) - len(coded) == len(NS))
check("No ARAM / Howling Abyss-only row remains in the Summoner's Rift coding",
      not coded.summary.astype(str).str.contains(r"^(?:ARAM|Howling Abyss)", regex=True).any())
sid, gno = assign_series(G)
check("Every game the source numbers 1 starts a series", (gno[G["game"] == 1] == 1).all())
in_tex("Worlds games", f"{len(G):,}".replace(",", "{,}") + " games")
in_tex("Team-years", f"{len(set(zip(G.year, G.blue_id)) | set(zip(G.year, G.red_id)))} team-years")
in_tex("Eligible team-years", f"{E.intended.notna().sum()} team-years")
n_out = int(PM.loc[('intended', 'd_intended_z'), 'n_games'])
in_tex("Outcome-model games", ("model includes all " if n_out == len(G) else "model includes ")
       + f"{n_out:,}".replace(",", "{,}") + " games")
check("All team-years have eligible pools", E.intended.notna().sum() == len(set(zip(G.year, G.blue_id)) | set(zip(G.year, G.red_id)))
      and f"All {E.intended.notna().sum()} team-years have at least ten official games" in TEX)
SA = json.loads((ROOT / "data/supplement/leaguepedia_agreement.json").read_text())
SL = SA["restored_by_league"]
in_tex("Restored games", f"{SA['games_restored']} games of that year's participants in the 365 days before Worlds")
check("Restored games: agreement rows are one-to-one", SA["shared_games_compared"] == SA["shared_games_unique_oe"])
in_tex("Restored games: agreement (3.1)", f"On the {SA['shared_games_compared']:,}".replace(",", "{,}") + " participants' games matched in both sources, "
       f"they agree on the winner in {100 * SA['agreement_winner']:.2f}\\% and on the bans in {100 * SA['agreement_bans_same_sets']:.1f}\\%")
groups = {"LPL": ["LPL"], "RR": ["RR"], "IWCQ": ["IWCQ"], "MSI": ["MSI"], "cups": ["NEST", "IEM", "DCup", "KeSPA", "SL ABCDE"]}
g = {k: sum(SL.get(x, 0) for x in v) for k, v in groups.items()}
g["other"] = SA["games_restored"] - sum(g.values())
in_tex("Restored games by competition", f"This adds {SA['games_restored']} games: {g['LPL']} from the LPL, {g['RR']} from Rift Rivals, "
       f"{g['other']} from other regional leagues and their cups, {g['cups']} from domestic and third-party cups (including Brazil's "
       f"Superliga ABCDE), {g['IWCQ']} from the "
       f"2016 International Wildcard Qualifier, and {words[g['MSI']]} from MSI. Of these, "
       f"{sum(v for k, v in SA['restored_by_year'].items() if int(k) <= 2019)} are from 2016--2019")
in_tex("Restored games: agreement (appendix)", "Leaguepedia's first-listed team is the blue side in " +
       ("all of them" if SA["agreement_team1_is_blue"] == 1 else f"{100 * SA['agreement_team1_is_blue']:.2f}\\%"))
VG = pd.read_csv(ROOT / "data/manifests/voided_games.csv")
L22 = set(load_matches(["gameid"], years=[2022]).gameid)
check("Voided games: excluded by the loader, replays kept",
      len(VG) == 3 and not (set(VG.gameid) & L22) and set(VG.replayed_as) <= L22
      and "three MSI 2022 games that Riot voided and replayed" in TEX)
in_tex("Restored games: patch agreement", f"Of the {SA['patch_compared']:,}".replace(",", "{,}") + " games for which both record a patch, "
       f"they agree on it in {100 * SA['agreement_patch']:.1f}\\%, and the {SA['patch_disagreements']} differences are each one version apart")
check("Patch differences are all one version apart", SA["patch_disagreements"] == SA["patch_disagreements_one_version_apart"])
TX = pd.read_csv(OUT / "team_exposure.csv", dtype={"last_patch": str, "last_patch_oe_only": str})
changed = TX[TX.last_patch != TX.last_patch_oe_only]
in_tex("Restored games: changed last versions", f"changes the last officially played version of {words[len(changed)]} team-years")
check("Restored games: changed teams named", all(n in TEX for n in changed.team))
in_tex("Restored games: changed pools", f"It also changes the 100-day pools of {int((TX.n_pool_games != TX.n_pool_games_oe_only).sum())} team-years")
in_tex("Restored games: newly eligible", f"{words[int(((TX.n_pool_games >= 10) & (TX.n_pool_games_oe_only < 10)).sum())]} team-years that had fewer "
       "than ten games in Oracle's Elixir now have eligible pools")
in_tex("Coded Summoner's Rift changes", f"leaving {len(coded):,}".replace(",", "{,}") + " changes on Summoner's Rift")

# ---------- 4.1 who pays ----------
D = json.loads((OUT / "manuscript_descriptives.json").read_text())
in_tex("Nerfed / buffed share of pre-tournament picks",
       f"{D['nerf_share_mean_pct']:.0f}\\% of a team's pre-Worlds picks were on champions with net nerfs in those patches "
       f"and {D['buff_share_mean_pct']:.0f}\\% on champions with net buffs")
in_tex("Within-year gaps", f"the median gap between the most and least affected teams was {D['nerf_share_median_range_pp']:.0f} percentage points in nerfed picks "
       f"and {D['buff_share_median_range_pp']:.0f} in buffed picks")
in_tex("Most-contested decile", f"{D['top_decile_nerf_pct']:.0f}\\% of champions were nerfed and "
       f"{D['top_decile_buff_pct']:.0f}\\% buffed")
M = pd.read_csv(OUT / "buff_persistence_mechanism.csv").set_index(["dep", "term"])
b, p = M.loc[("intended_z", "meta_z"), ["coef", "p"]]
check("Meta concentration -> net exposure: p < 0.001", p < 0.001)
in_tex("Meta concentration -> net exposure", f"{sgn(b, 2)} SD of net exposure ($p < 0.001$")
R = json.loads((OUT / "regional_champion_dispersion.json").read_text())
in_tex("Regional winners: median range", f"median of {R['median_range_net_z']:.2f} SD in net exposure")
in_tex("Random four teams", f"{R['random4_median_range_mean']:.2f} SD expected for four randomly drawn teams, whose median range has "
       "a 90\\% interval of {:.2f}--{:.2f}".format(*R["random4_median_range_90"]))
in_tex("Regional winners: meta-exposure correlation", f"with meta concentration spanning ${R['meta_z_min']:.1f}$ to ${R['meta_z_max']:+.1f}$ SD "
       f"($r = {R['corr_meta_net_regional']:.2f}$, against $r = {R['corr_meta_net_all']:.2f}$ for all eligible teams)")
RCM = pd.read_csv(ROOT / "data/manifests/regional_champions.csv").merge(
    E[["year", "teamid", "intended_z", "nerf_share"]], on=["year", "teamid"]).set_index(["year", "region"])
r23 = RCM.loc[2023].sort_values("nerf_share")
in_tex("Regional winners 2023", f"ranged from {100 * r23.nerf_share.iloc[0]:.0f}\\% ({r23.team.iloc[0]}) to "
       f"{100 * r23.nerf_share.iloc[-1]:.0f}\\% ({r23.team.iloc[-1]})")
in_tex("Least-contested deciles", f"In the bottom eight deciles, {D['bottom_eight_nerf_pct']:.0f}\\% were nerfed and "
       f"{D['bottom_eight_buff_pct']:.0f}\\% buffed")
bn, pn = M.loc[("nerfs_z", "meta_z"), ["coef", "p"]]
bb, pb = M.loc[("buffs_z", "meta_z"), ["coef", "p"]]
check("Meta concentration -> nerf exposure: p < 0.001", pn < 0.001)
in_tex("Meta concentration -> nerf and buff exposure", f"predicts {sgn(bn, 2)} SD of nerf exposure ($p < 0.001$), "
       f"{sgn(bb, 2)} SD of buff exposure ({pval(pb)})")
# dependence within a Worlds (two-way clusters, wild cluster bootstrap over Worlds)
INF = pd.read_csv(OUT / "buff_persistence_mechanism_inference.csv").set_index(["dep", "term"]).xs("meta_z", level="term")
check("Nerf and net associations: p <= 0.01 under Worlds-level inference",
      (INF.loc[["nerfs_z", "intended_z"], ["p_org_worlds_twoway_t9", "p_wild_cluster_worlds"]] <= 0.01).all().all()
      and "is allowed for ($p \\leq 0.01$)" in TEX)
check("Buff association is not robust (wild bootstrap p > 0.05), as 4.1 and appendix say",
      INF.loc["buffs_z", "p_wild_cluster_worlds"] > 0.05 and "Its association with buff exposure does not (Appendix" in TEX
      and "The buff association thus weakens under Worlds-level inference" in TEX)
in_tex("Worlds-level p-values (appendix)", "the two-way $p$-values are {:.3f}, {:.3f}, and {:.3f}, and the bootstrap $p$-values "
       "{:.3f}, {:.3f}, and {:.3f}".format(*INF.loc[["intended_z", "nerfs_z", "buffs_z"], "p_org_worlds_twoway_t9"],
                                            *INF.loc[["intended_z", "nerfs_z", "buffs_z"], "p_wild_cluster_worlds"]))
# presence reference
PS = pd.read_csv(OUT / "presence_reference_sensitivity.csv").set_index("reference")
ps = lambda ref, c: PS.loc[ref, c]
check("Presence: buff association disappears before the first change (p > 0.5), as the appendix says",
      ps("pre_change", "buffs_z_meta_p") > 0.5 and "it disappears when presence is measured before each year's first change" in TEX)
check("Presence: net and nerf associations hold under each definition",
      all(ps(r, "intended_z_meta_p") < 0.01 and ps(r, "nerfs_z_meta_p") < 0.01 for r in ["all", "major", "pre_change"]))
fp = lambda x: "$<$0.001" if x < 0.0005 else f"{x:.3f}"
cols = ["all", "major", "pre_change", "by_release"]
in_tex("Table (presence) rows", " & ".join(["Observations"] + [f"{int(ps(r, 'rows')):,}".replace(",", "{,}") for r in cols]) + " \\\\")
for lab, a, b_, dg in [("Top decile", "top_decile_nerf_pct", "top_decile_buff_pct", 0), ("Bottom eight", "bottom_eight_nerf_pct", "bottom_eight_buff_pct", 0)]:
    cells = [f"{ps(r, a):.{dg}f} / {ps(r, b_):.{dg}f}" for r in cols[:3]] + [f"{ps('by_release', a):.1f} / {ps('by_release', b_):.1f}"]
    in_tex(f"Table (presence) {lab}", f"{lab}: nerfed / buffed (\\%) & " + " & ".join(cells) + " \\\\")
for lab, dep in [("net", "intended_z"), ("nerf", "nerfs_z"), ("buff", "buffs_z")]:
    cells = [f"{sgn(ps(r, dep + '_meta_coef'), 2)} ({fp(ps(r, dep + '_meta_p'))})" for r in cols[:3]]
    in_tex(f"Table (presence) meta -> {lab}", f"Meta $\\to$ {lab} exposure & " + " & ".join(cells) + " & -- \\\\")

PT = pd.read_csv(OUT / "buff_persistence_tests.csv").set_index(["definition", "measure"])
PI = pd.read_csv(OUT / "buff_persistence_org_influence.csv")
pt = lambda d, m, c: PT.loc[(d, m), c]
LN, RC = "linked lineages", "recorded team IDs"
in_tex("Persistence, linked lineages", f"correlations across successive appearances are {pt(LN, 'breadth_z', 'next_appearance_r'):.2f} for breadth and "
       f"{pt(LN, 'meta_z', 'next_appearance_r'):.2f} for meta concentration, compared with "
       f"{pt(LN, 'buffs_z', 'next_appearance_r'):.2f} for buff exposure ($p = {pt(LN, 'buffs_z', 'p'):.2f}$, "
       f"{int(pt(LN, 'buffs_z', 'pairs'))} pairs)")
in_tex("Persistence, recorded team IDs", f"gives {pt(RC, 'breadth_z', 'next_appearance_r'):.2f}, "
       f"{pt(RC, 'meta_z', 'next_appearance_r'):.2f}, and {pt(RC, 'buffs_z', 'next_appearance_r'):.2f} "
       f"($p = {pt(RC, 'buffs_z', 'p'):.2f}$, {int(pt(RC, 'buffs_z', 'pairs'))} pairs)")
in_tex("Organization-variance test, both definitions", f"gives $p = {pt(RC, 'buffs_z', 'org_p'):.2f}$ before style adjustment and "
       f"$p = {pt(RC, 'buffs_z_res', 'org_p'):.3f}$ after it. Linking documented renames gives "
       f"$p = {pt(LN, 'buffs_z', 'org_p'):.2f}$ and $p = {pt(LN, 'buffs_z_res', 'org_p'):.3f}$")
pi = lambda o: PI[(PI.dropped_org == o) & (PI.measure == "buffs_z")].iloc[0]
in_tex("Organization-variance influence", f"averaged ${pi('DRX').mean_buffs_z:+.1f}$ SD of buff exposure, and without them "
       f"$p = {pi('DRX').org_p:.2f}$. Without T1, $p = {pi('T1').org_p:.3f}$")
CV = pd.read_csv(OUT / "champion_patch_vs_presence.csv")
CV = CV[CV.net != 0]
in_tex("Presence check", f"by {100 * CV[CV.net > 0].d_presence.mean():.1f}\\pp{{}} on average, and nerfed champions fell by "
       f"{-100 * CV[CV.net < 0].d_presence.mean():.1f}\\pp{{}} (correlation between coded score and presence change "
       f"$r = {np.corrcoef(CV.net, CV.d_presence)[0, 1]:.2f}$, $n = {len(CV)}$)")

# ---------- bans per team per game (Section 2) ----------
BN = load_matches(["gameid", "position", "ban1", "ban2", "ban3", "ban4", "ban5"], years=sorted(set(G.year)))
BN = BN[(BN.position == "team") & BN.gameid.isin(set(G.gameid))].merge(G[["gameid", "year"]].drop_duplicates(), on="gameid")
BN["n_bans"] = BN[["ban1", "ban2", "ban3", "ban4", "ban5"]].notna().sum(axis=1)
bans_max = BN.groupby("year").n_bans.max()
check("Bans per team per game at Worlds: at most three in 2016, five from 2017, and nearly always all used",
      bans_max.loc[2016] == 3 and (bans_max.drop(2016) == 5).all()
      and (BN.n_bans == BN.year.map(lambda y: 3 if y == 2016 else 5)).mean() > 0.99)
in_tex("Bans per team (Section 2)", "Each team can ban five champions per game, three in 2016, and picks five")

# ---------- 4.2 drafts ----------
B = pd.read_csv(OUT / "ban_pressure_targeting.csv").set_index(["model", "term"])
b1 = lambda term: B.loc[("B1 relief: other pool x buff exposure", term)]
g1 = lambda term: B.loc[("B1g1 series game 1", term)]
bb, bo = 100 * b1('pool_buff_x_expo').Estimate, 100 * b1('pool_other_x_expo').Estimate
check("Ban interaction: steeper for buffed champions, flatter for others", bb > 0 > bo)
in_tex("Ban interaction, buffed and other champions",
       f"makes this rise steeper by ${bb:.1f}$\\pp{{}} per unit of pick share for buffed champions ({pval(b1('pool_buff_x_expo')['Pr(>|t|)'])}) "
       f"and flatter by ${-bo:.1f}$\\pp{{}} for other champions ({pval(b1('pool_other_x_expo')['Pr(>|t|)'])})")
in_tex("Ban interaction, 10-point illustration",
       f"widens by about ${0.1 * bb:.1f}$\\pp{{}} ($0.1 \\times {bb:.1f}$\\pp{{}}) per SD of buff exposure")
in_tex("Ban interaction, series game 1",
       f"Restricted to series game 1, the rise for buffed champions is steeper by ${100 * g1('pool_buff_x_expo').Estimate:.1f}$\\pp{{}} "
       f"({pval(g1('pool_buff_x_expo')['Pr(>|t|)'])})")

TT = pd.read_csv(OUT / "ban_pressure_team_tests.csv")
fam = TT[(TT.iloc[:, 0] == "buffs_z") & (TT.iloc[:, 1] == "familiar")].iloc[0, 2]
in_tex("Familiar picks and buff exposure", f"the same role during the preceding 100 days ($r = {fam:.2f}$)")
WT = pd.read_csv(OUT / "ban_pressure_within_team.csv").set_index(["dep", "x"]).loc[("familiar", "pull")]
in_tex("Familiar picks within team", f"($\\beta = {WT.est:.3f}$, $p = {WT.p:.2f}$)")
in_tex("Ban model observations", "leaves " + f"{int(B.loc[('A0 pool targeting', 'pool'), 'n']):,}".replace(",", "{,}") + " observations")

# ---------- 4.3 what match results can establish ----------
net = PM.loc[("intended", "d_intended_z")]
in_tex("Net exposure and winning",
       f"{sgn(net.winprob_pp_per_sd)}\\pp{{}} in win probability per SD, with a 95\\% event-bootstrap interval from "
       f"{sgn(25 * net.boot_lo)} to {sgn(25 * net.boot_hi)} (studentized permutation {pval(net.p_perm)}")
buff = PM.loc[("nerfs + buffs", "d_buffs_z")]
in_tex("Buff exposure and winning", f"the buff coefficient is {sgn(buff.winprob_pp_per_sd)}\\pp{{}} (interval "
       f"{sgn(25 * buff.boot_lo)} to {sgn(25 * buff.boot_hi)}, {pval(buff.p_perm)})")
z = E.set_index(["year", "teamid"]).intended_z
Gm = G.assign(d=z.reindex(list(zip(G.year, G.blue_id))).to_numpy() - z.reindex(list(zip(G.year, G.red_id))).to_numpy())
Gm = Gm.dropna(subset=["d"])
fit = sm.Logit(Gm.result, sm.add_constant(Gm[["pred_logit", "d"]])).fit(disp=0)
check("Independent logit refit reproduces the net-exposure coefficient", abs(fit.params["d"] - net.estimate) < 1e-9)
check("Outcome-model game count matches the refit", len(Gm) == int(net.n_games))
S = pd.read_csv(OUT / "detectable_effect_summary.csv").set_index("scope")
check("Power summary uses the same permutation p", abs(S.loc["game", "p_perm"] - net.p_perm) < 1e-12)
cal = pd.read_csv(OUT / "permutation_null_calibration.csv")
cal = cal[(cal.statistic == "studentized") & (cal.n_sim == cal.n_sim.max())].iloc[0]
in_tex("Null rejection rate", f"rejected {100 * cal.rate:.1f}\\% of {int(cal.n_sim):,}".replace(",", "{,}") + " null simulations")
P = pd.read_csv(OUT / "detectable_effect_power.csv").set_index("winprob_pp_per_sd")
power = lambda pp: P.loc[P.index[np.isclose(P.index, pp)][0], "power"]
in_tex("Power at 2.5 and 3.5 pp", f"{100 * power(2.5):.0f}\\% at a positive $2.5$\\pp{{}} slope and "
       f"{100 * power(3.5):.1f}\\% at $3.5$\\pp{{}}")
in_tex("80% threshold, interpretation (4.3)", f"only per-game associations of about ${S.loc['game', 'mde80']:.1f}$\\pp{{}} per SD or larger")
in_tex("80% detection threshold per game", f"about {S.loc['game', 'mde80']:.1f}\\pp{{}} in either direction")
in_tex("Best-of-five illustration", f"roughly {S.loc['bo5', 'mde80']:.1f}\\pp{{}} in an evenly matched best-of-five")

# ---------- 4.4 cases ----------
in_tex("Most buff-exposed quarterfinalist", f"reached {words[D['most_buff_exposed_finals']]} of ten finals and won "
       f"{words[D['most_buff_exposed_titles']]} titles")
gov = json.loads((OUT / "governance_framing/summary.json").read_text())
lq, la = gov["lowest_net_quarterfinalists_finishes"], gov["lowest_net_all_eligible_finishes"]
in_tex("Lowest net exposure, quarterfinalists", f"{words[lq.get('QF', 0)].capitalize()} of these ten teams lost in the "
       f"quarterfinals, {words[lq.get('SF', 0)]} in the semifinals, and {words[lq.get('Runner-up', 0)]} reached the final and lost it")
BTC = pd.read_csv(OUT / "governance_framing/lowest_net_quarterfinalists.csv")
BTC = BTC[BTC.finish == "Champion"]
check("Lowest net exposure, quarterfinalists: titles", lq.get("Champion", 0) == len(BTC) == 1)
for r in BTC.itertuples():
    in_tex("Lowest net exposure, quarterfinalists: the title", f"One, {r.team} in {r.year}, won the title")
in_tex("Lowest net exposure, all participating teams", f"exited before the quarterfinals in {words[la.get('Before quarterfinals', 0)]} "
       f"of ten years and never won")
check("Lowest net exposure, all participating teams: no title", la.get("Champion", 0) == 0)
hb = pd.read_csv(OUT / "governance_framing/highest_buff_all_eligible.csv").finish.value_counts()
in_tex("Most buff-exposed, all participating teams", f"reached {words[hb.get('Runner-up', 0) + hb.get('Champion', 0)]} finals and won "
       f"{words[hb.get('Champion', 0)]} title, and {words[hb.get('Before quarterfinals', 0)]} of the ten exited")
KO = pd.read_csv(OUT / "knockout_patch_exposure.csv")
same = int((KO.loc[KO.groupby("year").intended.idxmin()].set_index("year").teamid
            == KO.loc[KO.groupby("year").nerfs.idxmax()].set_index("year").teamid).sum())
in_tex("Lowest net vs highest nerf exposure among quarterfinalists", f"identify the same team in {words[same]} of ten years")
BT = pd.read_csv(OUT / "governance_framing/lowest_net_quarterfinalists.csv")
for r in BT.itertuples():
    in_tex(f"Table (lowest-net quarterfinalists) {r.year}", f"{r.year} & {r.team} & ${r.intended_z:.2f}$ & {r.n_unseen_patches} &")

# ---------- ban-model specification checks (Appendix) ----------
spec = lambda model, term: B.loc[(model, term)]
for model, text in [("B1s1 exposure excluding the champion's own contribution", "gives"),
                    ("B1s2 + squared pool share", "gives"), ("B1s3 + buffed, nerfed, buffed x exposure", "gives")]:
    b_ = spec(model, "pool_buff_x_expo_loo" if model.startswith("B1s1") else "pool_buff_x_expo")
    in_tex(f"Ban specification {model[:4]}", f"{sgn(100 * b_.Estimate)}\\pp{{}} " + ("for buffed champions " if model.startswith("B1s1") else "")
           + f"({pval(b_['Pr(>|t|)'])}")

# ---------- roles (4.1, Appendix) ----------
RS = pd.read_csv(OUT / "role_exposure_summary.csv").set_index("role")
RJ = json.loads((OUT / "role_exposure_summary.json").read_text())
in_tex("Role nerf shares in 4.1", f"{RS.loc['bot', 'nerf_share_pct']:.0f}\\% of bottom-lane and {RS.loc['jng', 'nerf_share_pct']:.0f}"
       f"\\% of jungle picks were on champions with net nerfs, compared with {RS.loc['top', 'nerf_share_pct']:.0f}\\% in "
       f"top lane and {RS.loc['sup', 'nerf_share_pct']:.0f}\\% in support")
in_tex("Jungle or bottom lane as the sole largest role", f"In {RJ['sole_largest_counts']['jng'] + RJ['sole_largest_counts']['bot']} of the "
       f"{RJ['n_sole_largest']} team-years with a unique largest role, jungle or bottom lane had the largest absolute net exposure")
check("Role counts partition the team-years",
      RJ["n_sole_largest"] + RJ["n_tied_largest"] + RJ["n_zero_exposure"] == RJ["n_team_years"])
in_tex("Largest role share", f"Among the {RJ['n_share']} team-years with nonzero exposure, the role with the largest "
       f"absolute exposure accounted for a median of {100 * RJ['top_role_share_of_abs_net_median']:.0f}\\% of the sum of "
       "absolute role exposures (interquartile range {:.0f}--{:.0f}\\%)".format(*(100 * x for x in RJ["top_role_share_iqr"])))
in_tex("Tied and zero-exposure team-years", f"These counts cover {RJ['n_sole_largest']} team-years, excluding {RJ['n_tied_largest']} with tied "
       f"largest roles and {words[RJ['n_zero_exposure']]} with no exposure")
names = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bottom", "sup": "Support"}
for r, x in RS.iterrows():
    in_tex(f"Table (roles) {r}", f"{names[r]} & {x.nerf_share_pct:.1f} [{x.nerf_share_lo:.1f}, {x.nerf_share_hi:.1f}] & "
           f"{x.buff_share_pct:.1f} [{x.buff_share_lo:.1f}, {x.buff_share_hi:.1f}] & {RJ['sole_largest_counts'][r]} \\\\")

# ---------- Appendix A: coding validation ----------
H = pd.read_csv(ROOT / "paper/validation/human_vs_llm.csv")
sign = lambda d: np.where(d == "buff", 1, np.where(d == "nerf", -1, 0))
po = np.mean(sign(H.hd) == sign(H.ld))
pe = sum(np.mean(sign(H.hd) == k) * np.mean(sign(H.ld) == k) for k in (-1, 0, 1))
in_tex("Author direction agreement", f"Direction agreed in {(H.hd == H.ld).sum()} of the {len(H)} changes "
       f"({100 * np.mean(H.hd == H.ld):.1f}\\%)")
in_tex("Author sign agreement", f"in {(sign(H.hd) == sign(H.ld)).sum()} ({100 * po:.1f}\\%, $\\kappa = {(po - pe) / (1 - pe):.2f}$)")
GD = json.loads((OUT / "gamedata_direction_summary.json").read_text())
a, dd, cd = GD["all"], GD["Data Dragon 2016-18"], GD["CommunityDragon 2019-25"]
in_tex("Game-data agreement, all one-directional changes",
       f"matched the one-directional data in {100 * a['sign_match_one_directional']:.1f}\\% of the {a['n_one_directional']} "
       f"changes ({100 * dd['sign_match_one_directional']:.1f}\\% with Data Dragon, {100 * cd['sign_match_one_directional']:.1f}"
       f"\\% with CommunityDragon) and in {100 * a['sign_match_one_directional_strict']:.1f}\\% of the "
       f"{a['n_one_directional_strict']} without reshaped values")
in_tex("Game-data agreement, changes coded as buffs or nerfs", f"Among the {a['n_one_directional_coded_signed']} one-directional "
       f"changes coded as buffs or nerfs, the sign matched in {100 * a['sign_match_one_directional_coded_signed']:.1f}\\%")
in_tex("Game-data summary in 3.2", f"{100 * a['sign_match_one_directional']:.1f}\\% of {a['n_one_directional']} changes, and in "
       f"{100 * a['sign_match_one_directional_coded_signed']:.1f}\\% of the {a['n_one_directional_coded_signed']}")
in_tex("Game-data comparable changes", f"This left {a['n']} changes")
GR = pd.read_csv(OUT / "gamedata_direction_rows.csv")
in_tex("Game-data hotfix exclusion", f"also excluded {(GR.reason == 'hotfix in preceding patch').sum()} changes to champions "
       "hotfixed on Summoner's Rift during the preceding patch")
HX = pd.read_csv(ROOT / "data/patch_notes/hotfix_exclusions.csv", dtype={"patch": str})
check("Hotfix list: excluded entries are Summoner's Rift only", HX.loc[HX.exclude, "mode"].str.startswith("SR").all()
      and not HX.loc[~HX.exclude, "mode"].str.startswith("SR").any())
al, au = GD["author_sample_llm"], GD["author_sample_author"]
check("Author sample: LLM and author rates use the same changes", al["n_one_directional"] == au["n_one_directional"])
in_tex("Game-data, author sample", f"{al['n']} changes were comparable and {al['n_one_directional']} were one-directional. On "
       f"these same changes the coded sign matched in {100 * al['sign_match_one_directional']:.1f}\\% for the LLM codes and "
       f"{100 * au['sign_match_one_directional']:.1f}\\% for the author's codes")
X = pd.read_csv(OUT / "gamedata_direction_crosstab.csv", index_col=0)
label = {"buff": "Buff", "nerf": "Nerf", "adjust": "Adjustment", "rework": "Rework", "bugfix": "Bug fix"}
for k in X.index:
    in_tex(f"Table (game data) row {k}", f"{label[k]} & " + " & ".join(str(v) for v in X.loc[k]) + " \\\\")
C = pd.read_csv(ROOT / "output/coding_sensitivity/coding_sensitivity_summary.csv", index_col=0)
f = lambda row: [float(x) for x in C.loc[row]]
sp = lambda est, p: f"{sgn(est)} ({p:.3f})"
iv = lambda v: "[{}, {}]".format(*(sgn(float(x)) for x in v.split(" to ")))
table5 = {
    "Changes recoded": "& -- & " + " & ".join(C.iloc[0, 1:]),
    "Team net exposure, $r$ with main": " & ".join(["1"] + [f"{x:.2f}" for x in f("team net exposure: r with main coding")[1:]]),
    "Pre-Worlds picks nerfed / buffed": " & ".join(f"{n:.1f} / {b:.1f}" for n, b in zip(f("nerf share of pre-Worlds picks (%)"),
                                                                                     f("buff share of pre-Worlds picks (%)"))),
    "Top-decile champions nerfed / buffed": " & ".join(f"{n:.0f} / {b:.0f}" for n, b in zip(f("top-decile champions nerfed (%)"),
                                                                                           f("top-decile champions buffed (%)"))),
    "Meta concentration": " & ".join(sgn(x, 2) for x in f("meta concentration -> net exposure (SD)")),
    "Regional winners: median range": " & ".join(f"{x:.2f}" for x in f("regional champions: median net-exposure range (SD)")),
    "Regional winners: meta--exposure": " & ".join(sgn(x, 2) for x in f("regional champions: meta-exposure r")),
    "Ban interaction, buffed": " & ".join(sp(e, p) for e, p in zip(f("ban: buffed champions x exposure (pp)"), f("  p "))),
    "Ban interaction, other": " & ".join(sp(e, p) for e, p in zip(f("ban: other champions x exposure (pp)"), f("  p  "))),
    "Net exposure -> win": " & ".join(sgn(x) for x in f("win: net exposure (pp per SD)")),
    "interval": " & ".join(iv(v) for v in C.loc["  event-bootstrap interval (pp)"]),
    "permutation p": " & ".join(f"{x:.3f}" for x in f("  permutation p")),
    "null rejection rate": " & ".join(f"{100 * x:.1f}" for x in f("win: studentized test, null rejection rate")),
    "80% detection threshold": " & ".join(f"{x:.1f}" for x in f("win: 80% detection threshold (pp per game)")),
    "Buff exposure -> win": " & ".join(sp(e, p) for e, p in zip(f("win: buff exposure, buff+nerf model (pp per SD)"), f("  permutation p "))),
    "Most buff-exposed QF team": " & ".join(C.loc["most buff-exposed quarterfinalist: finals / titles"]),
    "Lowest-net QF team": " & ".join(C.loc["lowest-net quarterfinalist: finals / titles"]),
}
for name, cells in table5.items():
    in_tex(f"Table (coding sensitivity) {name}", cells + " \\\\")
KM = pd.read_csv(OUT / "knockout_patch_exposure.csv")
KB = pd.read_csv(ROOT / "output/coding_sensitivity/B_gamedata_directions/knockout_patch_exposure.csv")
top = lambda K: K.loc[K.groupby("year").buffs.idxmax()].set_index("year").team
low = lambda K: K.loc[K.groupby("year").intended.idxmin()].set_index("year").team
in_tex("Coding sensitivity: case teams changed under B", f"the identity of these teams changes in {words[int((top(KM) != top(KB)).sum())]} "
       f"and {words[int((low(KM) != low(KB)).sum())]} of the ten years")
bw = f("win: buff exposure, buff+nerf model (pp per SD)")
in_tex("Buff coefficient depends on coding (4.3)", "$p = 0.111$), but its size depends on how change directions are coded (Appendix")
GDR = pd.read_csv(OUT / "gamedata_direction_rows.csv", dtype={"patch": str})
CB = pd.read_csv(ROOT / "output/coding_sensitivity/B_gamedata_directions/coding.csv", dtype={"patch_oe": str})
check("Coding B file aligns row by row with the main coding",
      len(CB) == len(coded_all) and (CB.champion.values == coded_all.champion.values).all() and (CB.patch_oe.values == coded_all.patch_oe.values).all())
DB = sr_only(coded_all.reset_index(drop=True).join(CB[["direction"]].reset_index(drop=True), rsuffix="_B"))
lost = DB[(DB.direction == "buff") & (DB.direction_B != "buff")].merge(
    GDR.rename(columns={"year": "year_group", "patch": "patch_oe"})[["year_group", "patch_oe", "champion", "data_dir"]],
    on=["year_group", "patch_oe", "champion"], how="left", validate="one_to_one")
in_tex("Buffs removed or reversed under B (Appendix B)", f"Of the {len(lost)} buffs that B removes or reverses, "
       f"{int((lost.data_dir == 'none').sum())} show no change at all in the game data")
jv = GDR[(GDR.champion == "Jarvan IV") & (GDR.patch == "8.19")]
check("Jarvan IV 8.19: coded as a buff, no change visible in the Data Dragon files",
      len(jv) == 1 and jv.iloc[0].llm_dir == "buff" and jv.iloc[0].data_dir == "none" and jv.iloc[0].source == "Data Dragon")
check("Coding sensitivity: buff coefficient under A equals the main value to one decimal", f"{bw[1]:.1f}" == f"{bw[0]:.1f}")
in_tex("Coding sensitivity: buff coefficient under A and B", f"stays at {sgn(bw[1])}\\pp{{}} under A but falls to {sgn(bw[2])}\\pp{{}} under B")
in_tex("Coding sensitivity: threshold in text", f"stays near ${max(f('win: 80% detection threshold (pp per game)')):.1f}$\\pp{{}} per game")
check("Coding sensitivity: every outcome interval includes zero",
      all(float(v.split(" to ")[0]) < 0 < float(v.split(" to ")[1]) for v in C.loc["  event-bootstrap interval (pp)"]))
check("Coding sensitivity: main column equals the main run",
      abs(f("win: net exposure (pp per SD)")[0] - net.winprob_pp_per_sd) < 1e-9
      and abs(f("regional champions: meta-exposure r")[0] - R["corr_meta_net_regional"]) < 1e-9)

# ---------- pre-tournament strength (3.4 and appendix) ----------
RV = pd.read_csv(OUT / "rating_validation_game.csv")
rv = lambda i, c: RV.iloc[i][c]
RS = pd.read_csv(OUT / "rating_validation_series.csv").iloc[0]
RR = pd.read_csv(OUT / "rating_validation_region.csv").set_index("region")
in_tex("Rating accuracy (3.4)", f"It correctly classifies the winner in {100 * rv(3, 'acc'):.1f}\\% of Worlds games with a log loss of {rv(3, 'logloss'):.3f}, "
       f"or {rv(4, 'logloss'):.3f} when its two hyper-parameters, the half-life and the penalty on team deviations, "
       "are chosen leave-one-year-out")
RG = pd.read_csv(OUT / "rating_grid.csv")
RGV = pd.read_csv(OUT / "rating_validation_grid.csv")
rg_best = RG.sort_values("logloss_known_teams").iloc[0]
check("Rating grid: w1 and the LOYO validation search the same grid",
      set(zip(RG.half_life, RG.lam_team)) == set(zip(RGV.half_life, RGV.lam_team)))
fmt_list = lambda xs: ", ".join(f"${x:g}$" for x in xs[:-1]) + f", or ${xs[-1]:g}$"
in_tex("Rating grid (appendix)", f"the half-life of the down-weighting ({fmt_list(sorted(RG.half_life.unique()))} days) and the "
       f"penalty on team deviations ({fmt_list(sorted(RG.lam_team.unique()))}). The main rating uses the pair with the lowest log loss "
       f"over all Worlds games, ${rg_best.half_life:g}$ days and ${rg_best.lam_team:g}$")
check("Rating grid: the chosen penalty is not at the edge of the grid",
      RG.lam_team.min() < rg_best.lam_team < RG.lam_team.max() and RG.half_life.min() < rg_best.half_life < RG.half_life.max())
in_tex("Series aggregation", f"classifies {100 * RS.acc:.1f}\\% of series correctly. It underrates LCK teams by "
       f"{RR.loc['LCK', 'actual_minus_pred_pp']:.1f}\\pp{{}} per cross-region game")
in_tex("League effects only", f"league effects alone correctly classify {100 * rv(2, 'acc'):.1f}\\% of games")
cm = sm.Logit(G.result, sm.add_constant(G.pred_logit)).fit(disp=0)
ci = cm.conf_int().loc["pred_logit"]
in_tex("Calibration slope", f"calibration slope of {cm.params['pred_logit']:.2f} (95\\% CI {ci[0]:.2f}--{ci[1]:.2f})")
in_tex("Region bias", f"LCK teams won {RR.loc['LCK', 'actual_minus_pred_pp']:.1f}\\pp{{}} more than predicted (95\\% CI "
       f"{RR.loc['LCK', 'lo']:.1f}--{RR.loc['LCK', 'hi']:.1f}) and LPL teams {-RR.loc['LPL', 'actual_minus_pred_pp']:.1f}\\pp{{}} less "
       f"(CI ${RR.loc['LPL', 'lo']:.1f}$ to ${RR.loc['LPL', 'hi']:.1f}$)")
RT = pd.read_csv(OUT / "rating_validation_tournament.csv")
in_tex("Rating vs placement", f"averages {RT.spearman_rating_vs_finish.mean():.2f} per year, and the eventual winner's rating rank averages "
       f"{RT.champion_rating_rank.mean():.1f}")
for i, lab in [(0, "Coin flip"), (1, "Blue side only"), (2, "League effects only"), (3, "Rating (hyper-parameters chosen on all years)"),
               (4, "Rating (hyper-parameters chosen leave-one-year-out)")]:
    in_tex(f"Table (rating) {lab}", f"{lab} & {100 * rv(i, 'acc'):.1f}\\% & {rv(i, 'logloss'):.3f} \\\\")
SM = pd.read_csv(OUT / "strength_models.csv").set_index("control")
in_tex("90-day rating log loss", f"has worse log loss than a coin flip ({SM.loc['90d', 'logloss']:.3f} versus {rv(0, 'logloss'):.3f})")
HV = pd.read_csv(OUT / "historical_style_validation.csv").set_index("exposure").loc["intended_z"]
in_tex("Held-out style model: years in the right direction", f"in the right direction in {words[int(HV.loyo_spearman_n_positive)]} of the "
       f"{words[int(HV.loyo_years)]} years")
HY = pd.read_csv(OUT / "historical_style_validation_by_year.csv")
HY = HY[HY.exposure == "intended_z"].set_index("year").spearman
CPY = pd.read_csv(OUT / "buff_persistence_champion_presence.csv")
CPY["decile"] = np.minimum(np.ceil(CPY.pres_pct * 10).astype(int), 10)
top_nerf = CPY[CPY.decile == 10].groupby("year")["class"].apply(lambda c: 100 * (c == "nerfed").mean())
exc = HY[HY < 0]
check("Held-out style model: one exception year, and it is the year with the fewest top-decile nerfs",
      len(exc) == 1 and exc.index[0] == top_nerf.idxmin())
others = top_nerf.drop(exc.index[0])
in_tex("Held-out style model: the exception", f"The exception, {exc.index[0]}, is the year in which the late patches nerfed the fewest of the "
       f"most-contested champions (${top_nerf[exc.index[0]]:.0f}\\%$, against ${others.min():.0f}$--${others.max():.0f}\\%$ in other years)")
in_tex("Held-out style model: without the exception", f"Averaged over the other {words[len(HY) - len(exc)]} years, the correlation is "
       f"${HY.drop(exc.index).mean():.2f}$")
in_tex("Held-out style model", f"average Spearman correlation of {HV.loyo_spearman_mean:.2f} (range ${HV.loyo_spearman_min:.2f}$ to "
       f"${HV.loyo_spearman_max:.2f}$)")

# ---------- 4.2 ban decomposition, terciles, displacement ----------
TT = pd.read_csv(OUT / "ban_pressure_team_tests.csv").set_index(["a", "b"])
p2 = lambda p: "$p < 0.001$" if p < 0.001 else f"$p = {p:.2f}$"
tt = lambda b: TT.loc[("buffs_z", b)]
in_tex("Ban decomposition: removed share", f"Across the {int(tt('denied').n)} team-years with nonzero buff exposure, buff exposure correlates with the fraction of that exposure removed by opponent bans ($r = {tt('denied').r:.2f}$, {p2(tt('denied').p)})")
in_tex("Ban decomposition: expected", f"($r = {tt('denied_exp').r:.2f}$, {p2(tt('denied_exp').p)})")
in_tex("Ban decomposition: excess", f"($r = {tt('denied_excess').r:.2f}$, {p2(tt('denied_excess').p)})")
TC = pd.read_csv(OUT / "ban_pressure_terciles.csv").set_index("tercile")
in_tex("Ban terciles", f"faced {TC.loc['high', 'cb_obs']:.2f} opponent bans per game on buffed core champions against "
       f"{TC.loc['high', 'cb_exp']:.2f} expected, an excess of {TC.loc['high', 'pull']:.2f}")
check("Ban terciles: bottom third's excess approximately zero", abs(TC.loc["low", "pull"]) < 0.01)
in_tex("Ban terciles: other core champions", f"decreased from {TC.loc['low', 'other_pull']:.2f} to {TC.loc['high', 'other_pull']:.2f}")
WT = pd.read_csv(OUT / "ban_pressure_within_team.csv").set_index("dep").loc["other_pull"]
in_tex("Displacement within team", f"associated with {-WT.est:.2f} fewer on other core champions ({p2(WT.p)})")
SS = pd.read_csv(OUT / "strength_sensitivity.csv")
ss = SS[SS.control == "1y"].set_index("row")
in_tex("Excess bans and pick advantage -> win", f"have associations of {sgn(ss.loc['Excess bans', 'winprob_pp'])} and "
       f"{sgn(ss.loc['Pick advantage', 'winprob_pp'])}\\pp{{}} per SD, respectively, with intervals including zero")
check("Excess bans and pick advantage: intervals include zero", all(ss.loc[r, "lo"] < 0 < ss.loc[r, "hi"] for r in ["Excess bans", "Pick advantage"]))

# ---------- ban model appendix ----------
BT2 = pd.read_csv(OUT / "ban_pressure_targeting.csv")
BT2 = BT2.assign(e=100 * BT2.Estimate, lo=100 * BT2["2.5%"], hi=100 * BT2["97.5%"], p=BT2["Pr(>|t|)"]).set_index(["model", "term"])
bt = lambda m, t: BT2.loc[(m, t)]
in_tex("Ban appendix: baseline", f"a unit increase in pick share is associated with {sgn(bt('A0 pool targeting', 'pool').e)}\\pp{{}} in ban probability")
x = bt("A1 targeting x patch", "pool_nerf")
in_tex("Ban appendix: nerfed interaction", f"nerfed champions have a {sgn(x.e)}\\pp{{}} pick-share interaction ({pval(x.p)})")
b, o = bt("B1 relief: other pool x buff exposure", "pool_buff_x_expo"), bt("B1 relief: other pool x buff exposure", "pool_other_x_expo")
in_tex("Ban appendix: main intervals", f"yields {sgn(b.e)}\\pp{{}} for buffed champions (95\\% interval {sgn(b.lo)} to {sgn(b.hi)}) and "
       f"{sgn(o.e)}\\pp{{}} for other champions ({sgn(o.lo)} to {sgn(o.hi)})")
b, o = bt("B1x25 excl. 2025", "pool_buff_x_expo"), bt("B1x25 excl. 2025", "pool_other_x_expo")
in_tex("Ban appendix: excluding 2025", f"Excluding 2025 gives {sgn(b.e)}\\pp{{}} for buffed champions ({pval(b.p)}) and {sgn(o.e)}\\pp{{}} for others ({pval(o.p)})")
b, o = bt("B1g1 series game 1", "pool_buff_x_expo"), bt("B1g1 series game 1", "pool_other_x_expo")
in_tex("Ban appendix: game 1", f"Series game 1 gives {sgn(b.e)} and {sgn(o.e)}\\pp{{}} ($p = {b.p:.3f}$ and ${o.p:.3f}$)")
b, o = bt("B1p1 phase-1 bans", "pool_buff_x_expo"), bt("B1p1 phase-1 bans", "pool_other_x_expo")
in_tex("Ban appendix: first phase", f"First-phase bans give {sgn(b.e)} and {sgn(o.e)}\\pp{{}} ($p = {b.p:.3f}$ and ${o.p:.3f}$)")
o = bt("B1s1 exposure excluding the champion's own contribution", "pool_other_x_expo_loo")
in_tex("Ban appendix: B1s1 other", f"and {sgn(o.e)}\\pp{{}} for others ({pval(o.p)})")
b, o = bt("B1s2 + squared pool share", "pool_buff_x_expo"), bt("B1s2 + squared pool share", "pool_other_x_expo")
in_tex("Ban appendix: B1s2", f"gives {sgn(b.e)}\\pp{{}} ({pval(b.p)}; 95\\% interval {sgn(b.lo)} to {sgn(b.hi)}) and {sgn(o.e)}\\pp{{}} ({pval(o.p)})")
b3, o = bt("B1s3 + buffed, nerfed, buffed x exposure", "pool_buff_x_expo"), bt("B1s3 + buffed, nerfed, buffed x exposure", "pool_other_x_expo")
in_tex("Ban appendix: B1s3", f"gives {sgn(b3.e)}\\pp{{}} ({pval(b3.p)}) and {sgn(o.e)}\\pp{{}} ({pval(o.p)})")
es = [bt(m, t).e for m, t in [("B1s1 exposure excluding the champion's own contribution", "pool_buff_x_expo_loo"),
                              ("B1s2 + squared pool share", "pool_buff_x_expo"), ("B1s3 + buffed, nerfed, buffed x exposure", "pool_buff_x_expo")]]
in_tex("Ban appendix: specification range", f"its size ranges from {sgn(min(es))} to {sgn(max(es))}\\pp{{}} across specifications")
check("Ban 4.2: every specification is positive", min(es) > 0)
in_tex("Ban 4.2: specification range", f"it stays steeper, by ${min(es):.1f}$ to ${max(es):.1f}$\\pp{{}} (Appendix")
in_tex("Ban appendix: B1s2 interval", f"($p = {bt('B1s2 + squared pool share', 'pool_buff_x_expo').p:.3f}$; 95\\% interval "
       f"{sgn(bt('B1s2 + squared pool share', 'pool_buff_x_expo').lo)} to {sgn(bt('B1s2 + squared pool share', 'pool_buff_x_expo').hi)})")

# ---------- power appendix ----------
CAL = pd.read_csv(OUT / "permutation_null_calibration.csv").set_index(["statistic", "n_sim"])
cs2, cr2 = CAL.loc[("studentized", 2000)], CAL.loc[("raw", 2000)]
in_tex("Power appendix: null calibration", f"the rejection rate was {100 * cs2.rate:.1f}\\% (95\\% Monte Carlo interval "
       f"{100 * cs2.mc_lo:.1f}--{100 * cs2.mc_hi:.1f}\\%). The unstudentized coefficient test rejected {100 * cr2.rate:.1f}\\%")
PWR = pd.read_csv(OUT / "detectable_effect_power.csv")
pw = lambda v: PWR[(PWR.winprob_pp_per_sd - v).abs() < 1e-9].iloc[0]
in_tex("Power appendix: power", f"power is {100 * pw(2.5).power:.0f}\\% at a positive $2.5$\\pp{{}} slope and {100 * pw(3.5).power:.1f}\\% at $3.5$\\pp{{}}")
in_tex("Power appendix: Monte Carlo intervals", f"are {100 * pw(2.5).mc_lo:.1f}--{100 * pw(2.5).mc_hi:.1f}\\% and "
       f"{100 * pw(3.5).mc_lo:.1f}--{100 * pw(3.5).mc_hi:.1f}\\%")
in_tex("Power appendix: thresholds", f"Interpolated 80\\% thresholds are approximately ${S.loc['game', 'mde80']:.1f}$\\pp{{}} in either direction")
check("Power appendix: thresholds agree in both directions to one decimal",
      abs(S.loc["game", "mde80"] - S.loc["game", "mde80_negative"]) < 0.1)
in_tex("Power appendix: best-of-five", f"best-of-five series gives about ${S.loc['bo5', 'mde80']:.1f}$\\pp{{}}")

# ---------- unavailable champions (3.5) ----------
AV = pd.read_csv(ROOT / "data/manifests/champion_availability.csv")
AV = AV[AV.scope != "uncertain"]
av_kind = AV.reason.str.split(":").str[0].value_counts()
in_tex("Unavailable champions (3.5)", f"Riot disabled {words[len(AV)]} champions for all or part of a Worlds. "
       f"{words[int(av_kind.get('policy_disable', 0))].capitalize()} were new or reworked champions released too close to the event, "
       f"and {words[int(av_kind.get('bug_disable', 0))]} were disabled for bugs")
check("Unavailable champions: one row per champion-year", not AV.duplicated(["year", "champion"]).any())

# ---------- organization lineages (3.5 and Appendix E) ----------
OL = pd.read_csv(ROOT / "data/manifests/org_lineage.csv")
in_tex("Lineages (3.5)", f"{words[OL.lineage_id.nunique()]} lineages listed with sources in Appendix")
in_tex("Lineages (Appendix E)", f"lists the {words[OL.lineage_id.nunique()]} lineages")
lin_rows = re.findall(r"\n([^\n&]*\\rightarrow[^\n&]*) & [^\n]* & [0-9][^\n]*\\\\", TEX[TEX.index("label{tab:lineages}"):TEX.index("end{tabular}", TEX.index("label{tab:lineages}"))])
check("Lineage table: one row per lineage in the manifest", len(lin_rows) == OL.lineage_id.nunique())

# ---------- abstract and other restatements ----------
in_tex("Abstract: pick shares", f"On average, {D['nerf_share_mean_pct']:.0f}\\% of a team's prior picks were on champions with net nerfs in the patches "
       f"it had not played before Worlds, and {D['buff_share_mean_pct']:.0f}\\% were on champions with net buffs. Within a year, "
       f"the median gap between the most and least affected teams was {D['nerf_share_median_range_pp']:.0f} percentage points for nerfed picks "
       f"and {D['buff_share_median_range_pp']:.0f} for buffed picks")
in_tex("Abstract: outcome", f"at {sgn(net.winprob_pp_per_sd)} percentage points per standard deviation with a 95\\% interval from "
       f"{sgn(25 * net.boot_lo)} to {sgn(25 * net.boot_hi)}")
tc = lambda n: f"{n:,}".replace(",", "{,}")
lq_finals = lq.get("Runner-up", 0) + lq.get("Champion", 0)
in_tex("Abstract: quarterfinal cases", f"the most buff-exposed team of each year reached {words[D['most_buff_exposed_finals']]} of ten finals "
       f"and won {words[D['most_buff_exposed_titles']]} titles. The team with the most negative net exposure reached "
       f"{words[lq_finals]} finals and won {words[lq.get('Champion', 0)]}")
in_tex("Abstract: sample", f"combining {tc(len(G))} games, {E.intended.notna().sum()} team-years, and LLM coding of "
       f"{tc(len(coded))} champion changes across {coded.patch_oe.nunique()} patches")
in_tex("Window length", f"ranges from {words[D['unseen_min']]} to {words[D['unseen_max']]} (mean {D['unseen_mean']:.1f})")
in_tex("Outcome n", f"studentized permutation {pval(net.p_perm)}, $n = " + f"{int(net.n_games):,}".replace(",", "{,}") + "$")
in_tex("Threshold in 4.3", f"with an interpolated 80\\% threshold of about ${S.loc['game', 'mde80']:.1f}$\\pp{{}} in either direction")
bt_recent = BT.set_index("year")
in_tex("Recent burden cases", "The recent cases were {} in 2023 (${:.2f}$ SD), {} in 2024 (${:.2f}$ SD), and {} in 2025 (${:.2f}$ SD)".format(
    bt_recent.loc[2023, "team"].replace("Weibo Gaming", "Weibo Gaming"), bt_recent.loc[2023, "intended_z"],
    bt_recent.loc[2024, "team"], bt_recent.loc[2024, "intended_z"], bt_recent.loc[2025, "team"], bt_recent.loc[2025, "intended_z"]))
in_tex("Figure 2a coverage", "covers " + f"{D['champion_years']:,}".replace(",", "{,}") + " champion-years with nonzero presence")
in_tex("Figure 2 caption: teams", f"Net exposure against meta concentration for {E.intended.notna().sum()} team-years")
in_tex("Restored games: limitations", f"The {SA['games_restored']} games restored from Leaguepedia cannot themselves be checked")
in_tex("Restored games: data availability", f"The {SA['games_restored']} games restored from Leaguepedia \\citep{{leaguepedia}} are included in the repository")
in_tex("Restored games: agreement sentence (appendix)", f"On the {SA['shared_games_compared']:,}".replace(",", "{,}") +
       " participants' games matched in both sources, Leaguepedia's first-listed team is the blue side in " +
       ("all of them" if SA["agreement_team1_is_blue"] == 1 else f"{100 * SA['agreement_team1_is_blue']:.2f}\\%") +
       f", and the sources agree on the winner in {100 * SA['agreement_winner']:.2f}\\% and on the set of bans in "
       f"{100 * SA['agreement_bans_same_sets']:.1f}\\%")

# ---------- corrections to Oracle's Elixir records (w0a, w0c) ----------
OC = pd.read_csv(ROOT / "data/manifests/oe_corrections.csv")
RC = pd.read_csv(ROOT / "data/manifests/oe_record_checks.csv")
iss, act = OC.issue.astype(str), OC.action
plain = int((iss.str.startswith("duplicate:") & (act == "drop")).sum())
mixcopy = int((iss.str.startswith("mixed:") & (act == "drop")).sum())
amb = OC[iss == "duplicate records naming different team pairs"]
nwin = OC[iss == "no winner recorded"]
dis = OC[iss == "OE and Leaguepedia winners differ"]
mixk = OC[iss == "team rows from another game"]
scopy = OC[iss == "statistics copied from another game"]
relab = OC[(iss == "team label") & (act == "relabel")]
nw_ldl18 = int(((nwin.league == "LDL") & nwin.date.astype(str).str.startswith("2018")).sum())
dropped = int((act == "drop").sum()); setw = int((act == "set_winner").sum()); excl = int((act == "exclude").sum())
check("Corrections: one row per removed or excluded game", OC.loc[act.isin(["drop", "exclude"]), "gameid"].is_unique)
in_tex("Corrections (3.1): the two-source rule is stated for missing winners only",
       "A missing winner is set only when at least two sources agree and none contradicts. These sources are the end-of-game "
       "statistics in Oracle's Elixir, Leaguepedia, a Games of Legends game page \\citeyearpar{golgg}, and a Liquipedia series "
       "score \\citeyearpar{liquipedia}. Games whose Oracle's Elixir and Leaguepedia winners differ are excluded")
check("Corrections: every winner set comes from a game without a recorded winner, every conflict is excluded",
      set(OC.loc[act == "set_winner", "issue"]) <= {"no winner recorded"} and (dis.action == "exclude").all())
check("Outcome conclusion (4.3): interval includes zero, stated as such",
      PM.loc[('intended', 'd_intended_z')].boot_lo < 0 < PM.loc[('intended', 'd_intended_z')].boot_hi
      and "Our outcome analysis establishes neither a causal advantage from late patches nor its absence, because the observed interval includes zero" in TEX)
n_amb_groups = RC.loc[RC.action == "resolve teams in w0c", "group"].nunique()
sc = RC[RC.check == "stats_copy"]
mixed_ids = set(RC.loc[RC.check == "mixed", "gameid"])
rows_t = [f"Copies naming the same teams & keep one record & {plain} removed \\\\",
          f"Copies carrying another game's team rows & keep the record whose team rows fit its players & {mixcopy} removed \\\\",
          f"Copies naming different team pairs ({n_amb_groups} groups, from Benelux, Northern European, and Taiwanese competitions) & keep the "
          f"record whose pair Leaguepedia confirms & {int((amb.action == 'keep').sum())} kept, {int((amb.action == 'drop').sum())} removed \\\\",
          f"Statistics shared across drafts ({sc.group.nunique()} groups, {len(sc)} records) & keep only when Leaguepedia records the same winner "
          f"& {int((scopy.action == 'keep').sum())} kept \\\\",
          f"Winner differs from Leaguepedia ({len(dis)} games) & sources contradict & {int((dis.action == 'exclude').sum())} excluded \\\\",
          f"No winner recorded ({len(nwin)} games, {nw_ldl18} of them from the 2018 LDL finals) & set when two sources agree and none contradicts "
          f"& {int((nwin.action == 'set_winner').sum())} set, {int((nwin.action == 'exclude').sum())} excluded \\\\",
          f"Team rows from another game ({len(mixk)} records with a recorded winner) & keep without bans when Leaguepedia, the result, and the "
          f"statistics agree & {int((mixk.action == 'keep_ignore_team_bans').sum())} kept, {int((mixk.action == 'exclude').sum())} excluded \\\\",
          f"& {len(relab)} moved \\\\",
          f"In total, {dropped} records are removed, {setw} winners set, {excl} games excluded, and {len(relab)} records moved"]
for k, t in enumerate(rows_t):
    in_tex(f"Table (corrections) row {k + 1}", t)
check("Corrections: ambiguous groups come from those competitions",
      set(RC.loc[RC.action == "resolve teams in w0c", "league"]) <= {"ESLOL", "OTBLX", "NLC", "PCL"})
check("Corrections: every disputed winner excluded", (dis.action == "exclude").all())
check("Corrections: winners set only for games without a recorded winner", set(OC.loc[act == "set_winner", "issue"]) == {"no winner recorded"})
check("Corrections: table rows add up to the totals", plain + mixcopy + int((amb.action == "drop").sum()) == dropped
      and int((dis.action == "exclude").sum()) + int((nwin.action == "exclude").sum()) + int((mixk.action == "exclude").sum())
      + int((scopy.action == "exclude").sum()) == excl)
OS = json.loads((ROOT / "data/manifests/oe_corrections_summary.json").read_text())
in_tex("Corrections: statistics rule consistency", "the statistics rule names a side in "
       + f"{OS['stats_rule_conclusive']:,}".replace(",", "{,}") + " of the " + f"{OS['stats_rule_games_one_winner']:,}".replace(",", "{,}")
       + f" Oracle's Elixir games of 2016--2025 with one recorded winner and agrees with that winner in {OS['stats_rule_agree_pct']:.2f}\\%")
check("Corrections: summary counts match the corrections file",
      OS["actions"] == {f"{i} | {a}": int(n) for (i, a), n in OC.assign(kind=iss.str.split(":").str[0]).groupby(["kind", "action"]).size().items()})
bds = relab[relab.league.eq("CDF") & (relab.blue.eq("Team BDS") | relab.red.eq("Team BDS"))]
in_tex("Corrections: team labels, BDS", f"{words[len(bds)].capitalize()} of the relabeled records are 2023 Coupe de France games of Team BDS "
       "Academy recorded as Team BDS")
check("Corrections: relabelled BDS games carry Team BDS Academy", (bds.new_teamname == "Team BDS Academy").all() and len(bds) > 0)
in_tex("Corrections: no Worlds game (3.1 and appendix)", "No correction touches a Worlds game")
check("Corrections: no Worlds game touched", not set(OC.gameid) & set(G.gameid))
# the corrections as the analyses see them (the loader, not the manifest)
LM = load_matches(["gameid", "date", "position", "side", "teamid", "result"])
LT = LM[LM.position == "team"]
check("Loader: no removed or excluded game remains", not set(OC.loc[act.isin(["drop", "exclude"]), "gameid"]) & set(LM.gameid))
won = LT[LT.result == 1].set_index("gameid").teamid
sw = OC[act == "set_winner"]
check("Loader: every winner set is applied", all(won.get(g) == t for g, t in zip(sw.gameid, sw.winner_teamid)))
side_team = LT.set_index(["gameid", "side"]).teamid
check("Loader: every relabel is applied", all(side_team.get((g, sd)) == t for g, sd, t in zip(relab.gameid, relab.side, relab.new_teamid)))
check("Loader: one winner per game", (LT.groupby("gameid").result.sum() == 1).all())
# participants' games removed from their 365-day history by an exclusion (copies are not games)
EVM = pd.read_csv(ROOT / "data/manifests/worlds_events.csv", index_col="year", parse_dates=["start"])
raw_t = pd.concat([pd.read_csv(f, usecols=["gameid", "date", "position", "teamid"], low_memory=False)
                   for f in sorted((ROOT / "data").glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))])
lost_ids = set(OC.loc[act == "exclude", "gameid"]) | set(OC.loc[(act == "drop") & (OC.reason == "no record's pair confirmed"), "gameid"])
raw_t = raw_t[(raw_t.position == "team") & raw_t.gameid.isin(lost_ids)]   # excluded games and copy groups dropped whole
raw_t["date"] = pd.to_datetime(raw_t.date)
part = set(zip(G.year, G.blue_id)) | set(zip(G.year, G.red_id))
lost = sorted({r.gameid for r in raw_t.itertuples() for y in EVM.index
               if (y, r.teamid) in part and EVM.loc[y, "start"] - pd.Timedelta(days=365) <= r.date < EVM.loc[y, "start"]})
check(f"Participants' games lost to exclusions within their window are the one named in appendix A: {lost}", lost == ["1364-1428"])
in_tex("Absent games named", "game 1 of the 2016 Demacia Cup final, won by EDward Gaming according to Leaguepedia")
RS = json.loads((OUT / "roster_scan_summary.json").read_text())
fmt = lambda n: f"{n:,}".replace(",", "{,}")
in_tex("Roster scan", f"A scan of the {fmt(RS['participant_games'])} games that participants played in the 365 days before Worlds finds "
       f"{RS['flagged']} in which a participant used at most one of its regular players ({RS['flagged_team_games']} participant--game records)")
in_tex("Roster scan by source", f"For the {RS['flagged_oe_games']} games ({RS['flagged_oe_team_games']} records) from Oracle's Elixir, the team "
       f"agrees with the matched Leaguepedia game. The other {RS['flagged_restored_games']} ({RS['flagged_restored_team_games']} records) "
       "were restored from Leaguepedia and keep its labels")
check("Roster scan: every flagged Oracle's Elixir game has Leaguepedia's team",
      RS["oe_leaguepedia_same_team"] == RS["flagged_oe_team_games"] and RS["oe_leaguepedia_other_team"] == 0 and RS["oe_not_in_leaguepedia"] == 0)

# ---------- sources of the most buff-exposed quarterfinalist's exposure, 2022-2025 (4.4; w29) ----------
BS = json.loads((OUT / "buff_sources_summary.json").read_text())
BR = pd.read_csv(OUT / "buff_sources.csv")
KO = pd.read_csv(OUT / "knockout_patch_exposure.csv")
YRS = [2022, 2023, 2024, 2025]
check("Buff sources: one row per quarterfinalist, eight per year",
      not BR.duplicated(["year", "teamid"]).any() and set(zip(BR.year, BR.teamid)) == set(zip(KO.year, KO.teamid))
      and (BR.groupby("year").size() == 8).all())
TPE = BR.merge(E[["year", "teamid", "buffs"]], on=["year", "teamid"])
check("Buff sources: decomposition adds up to the buff exposure of 3.3", len(TPE) == len(BR) and np.allclose(TPE.buff_exposure, TPE.buffs, atol=1e-9))
case = BR[(BR.buff_rank_qf == 1) & BR.year.isin(YRS)].sort_values("year")
check("Buff sources: T1 is the most buff-exposed quarterfinalist in each of 2022-2025", list(case.year) == YRS and (case.team == "T1").all())
if list(case.year) != YRS:                      # later checks need all four years; the failure above is reported
    YRS = list(case.year)
# ranks recomputed from w17's team table, not taken from w29
BT_ = pd.read_csv(OUT / "buff_persistence_teams.csv").merge(KO[["year", "teamid"]], on=["year", "teamid"])
rank_of = lambda col, asc: BT_.assign(r=BT_.groupby("year")[col].rank(ascending=asc, method="min")).set_index(["year", "teamid"]).r
t1ids = list(zip(case.year, case.teamid))
adj = [int(rank_of("buffs_z_res", False)[k]) for k in t1ids]
brd = [int(rank_of("breadth", False)[k]) for k in t1ids]
met = [int(rank_of("meta", True)[k]) for k in t1ids]
check("Buff sources: adjusted, breadth and meta ranks agree with w17 and the summary",
      adj == [BS["adjusted_rank_by_year"][str(y)] for y in YRS] and brd == [1, 1, 1, 1]
      and met == [BS["meta_rank_from_least_by_year"][str(y)] for y in YRS])
in_tex("Buff sources: breadth and meta ranks", "T1's pools were the broadest among quarterfinalists in 2022--2025, and their ranks from least to "
       "most meta-concentrated were " + ", ".join(map(str, met[:-1])) + f", and {met[-1]}")
ordn = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
in_tex("Buff sources: adjusted ranks", (f"T1 ranked {ordn[adj[0]]} among quarterfinalists in 2022 and 2025, {ordn[adj[1]]} in 2023, "
       f"and {ordn[adj[2]]} in 2024") if len(adj) == 4 else "(four years required)")
check("Buff sources: 2022 and 2025 adjusted ranks equal", len(adj) == 4 and adj[0] == adj[-1])
# shares recomputed from the per-team rows
w_ = case.buff_exposure / case.buff_exposure.sum()
pooled20 = float((w_ * case.pre_presence_20_plus_pct).sum())
check("Buff sources: pooled share matches the summary", abs(pooled20 - BS["pooled_pre_presence_20_plus_pct"]) < 1e-9)
check("Buff sources: no part from the most-contested decile", (case.top_decile_pct == 0).all() and BS["top_decile_pct_max"] == 0)
in_tex("Buff sources: top decile", "None of T1's buff exposure came from the most-contested decile of pre-Worlds presence (Section")
in_tex("Buff sources: pooled presence", f"weighted by raw buff exposure, {pooled20:.0f}\\% came from champions with at least 20\\% presence in "
       f"all recorded games during the 100 days before the year's first coded patch was released. This share ranged from "
       f"{case.pre_presence_20_plus_pct.min():.0f}\\% to {case.pre_presence_20_plus_pct.max():.0f}\\% across years")
in_tex("Buff sources: shared", f"In each year, {case.shared_pct.min():.0f}--{case.shared_pct.max():.0f}\\% came from champions also "
       "net-buffed in at least one other quarterfinalist's pool")
y22 = case[case.year == 2022].iloc[0]
check("Buff sources: 2022 top champion unique", ";" not in str(y22.top_champions))
in_tex("Buff sources: 2022 example", f"in 2022, {y22.top_champions} made up {y22.top_champion_role_share_pct:.0f}\\% of T1's "
       f"{dict(jng='jungle', top='top-lane', mid='mid-lane', bot='bottom-lane', sup='support')[y22.top_champion_role]} picks and "
       f"{y22.top_champion_pct:.0f}\\% of its buff exposure across roles")
check("Buff sources: the unsupported explanations are gone",
      "ranked first largely because" not in TEX and "explain only part of the ranking" not in TEX and "expected consequence" not in TEX)

# ---------- report ----------
failed = [n for n, ok in results if not ok]
for n, ok in results:
    print(("ok    " if ok else "FAIL  ") + n)
print(f"\n{len(results) - len(failed)} of {len(results)} manuscript numbers match the outputs")
sys.exit(1 if failed else 0)
