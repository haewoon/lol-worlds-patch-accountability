"""English figures for the arXiv preprint (vector PDF + PNG preview) -> paper/figures/.

Fig 1  three levels: pool exposure (buffed | nerfed share), targeted bans by buff tercile, result vs detection limit
Fig 2  mechanism: which champions get nerfed/buffed by pro presence; team meta concentration vs net exposure
Fig A1 result estimates under alternative pre-Worlds strength controls
Inputs are the versioned corrected outputs (w8, w10, w13-w17).
"""
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "worlds"))
from worlds_common import OUT
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, QF = "#2a78d6", "#eb6834", "#9ec5f4"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "pdf.fonttype": 42, "axes.unicode_minus": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})
SHORT = {"SK Telecom T1": "SKT", "Samsung Galaxy": "SSG", "Invictus Gaming": "IG", "FunPlus Phoenix": "FPX",
         "Dplus Kia": "DWG", "EDward Gaming": "EDG", "Kiwoom DRX": "DRX", "T1": "T1"}
KO = pd.read_csv(OUT / "knockout_patch_exposure.csv")
CH = KO[KO.finish == "Champion"].set_index(["year", "teamid"]).team
KO8 = set(zip(KO.year, KO.teamid))
REG = pd.read_csv(ROOT / "data" / "manifests" / "regional_champions.csv")      # LCK, LPL, LEC, LCS/LTA winners
REG8 = set(zip(REG.year, REG.teamid))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", color=INK, pad=8)


# ---------- Figure 1: three levels (print size, 6.5 in) ----------
XB, XN = 33, 65                                   # widest buffed share: FunPlus Phoenix 2019, 30.6%
fig = plt.figure(figsize=(6.5, 6.0))
g1 = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.94, bottom=0.575, wspace=0.05, width_ratios=[XB, XN])
g2 = fig.add_gridspec(1, 2, left=0.105, right=0.985, top=0.43, bottom=0.08, wspace=0.62)
axes = [fig.add_subplot(g1[0]), fig.add_subplot(g1[1]), fig.add_subplot(g2[0]), fig.add_subplot(g2[1])]

L1 = pd.read_csv(OUT / "team_patch_exposure.csv").dropna(subset=["buff_share"])
L1["champ"] = [(y, t) in CH.index for y, t in zip(L1.year, L1.teamid)]
L1["qf"] = pd.Series([(y, t) in KO8 for y, t in zip(L1.year, L1.teamid)], index=L1.index) & ~L1.champ
L1["reg"] = pd.Series([(y, t) in REG8 for y, t in zip(L1.year, L1.teamid)], index=L1.index)
assert L1.reg.sum() == len(REG)
years = sorted(L1.year.unique())
yi = L1.year.map({y: i for i, y in enumerate(years)})
for ax, col, xmax, xlab in [(axes[0], "buff_share", XB, "Buffed picks (%)"),
                            (axes[1], "nerf_share", XN, "Nerfed picks (%)")]:
    v = L1[col] * 100
    oth = ~L1.champ & ~L1.qf
    rg = L1[L1.reg]                                           # span of the four regional winners in each year
    ax.hlines(yi[rg.index].groupby(rg.year).first(), v[rg.index].groupby(rg.year).min(), v[rg.index].groupby(rg.year).max(),
              color=S2, lw=1.1, alpha=0.45, zorder=1.5)
    ax.scatter(v[oth], yi[oth], s=9, color=MUTED, alpha=0.45, edgecolor="white", linewidth=0.5, zorder=2, label="Other teams")
    ax.scatter(v[L1.qf], yi[L1.qf], s=11, color=QF, edgecolor="white", linewidth=0.5, zorder=3, label="Quarterfinalists")
    ax.scatter(v[L1.champ], yi[L1.champ], s=20, color=S1, edgecolor="white", linewidth=0.8, zorder=4, label="Worlds winner")
    ax.scatter(v[L1.reg], yi[L1.reg], s=38, facecolor="none", edgecolor=S2, linewidth=1.0, zorder=5,
               label="Regional winner")
    for i, r in L1[L1.champ].iterrows():
        ax.annotate(SHORT.get(CH[(r.year, r.teamid)], ""), (v[i], yi[i]), xytext=(0, 3.5), textcoords="offset points",
                    ha="center", fontsize=5.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_xticks(range(0, xmax, 10))
    ax.set_ylim(len(years) - 0.4, -1.9)
    ax.set_yticks(range(len(years)), [str(y) for y in years])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlab)
axes[1].tick_params(labelleft=False)
axes[1].legend(loc="upper right", ncol=4, frameon=False, fontsize=6.5, labelcolor=INK2, handletextpad=0.1, columnspacing=0.6)
panel_title(axes[0], "a", "Pool: the same patch lands differently")

ax = axes[2]
BS = pd.read_csv(OUT / "ban_pressure_team_summary.csv")
BS["ter"] = BS.groupby("year").buffs_z.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
for col, colr, lab, dx in [("pull", S1, "Buffed core champions", -0.08), ("other_pull", S2, "Other core champions", 0.08)]:
    m, lo, hi = [], [], []
    for k in range(3):
        v = BS.loc[BS.ter == k, col].dropna().to_numpy()
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        m.append(v.mean()); lo.append(np.quantile(bs, .025)); hi.append(np.quantile(bs, .975))
    x = np.arange(3) + dx
    ax.vlines(x, lo, hi, color=colr, lw=1.4, alpha=0.45, zorder=2)
    ax.plot(x, m, color=colr, lw=1.5, marker="o", ms=4.5, mec="white", mew=1, zorder=3, label=lab)
ax.axhline(0, color=AXIS, lw=1, zorder=1)
ax.set_xticks(range(3), ["Bottom third", "Middle", "Top third"])
ax.set_xlim(-0.4, 2.4)
ax.set_ylim(-0.08, 0.85)
ax.grid(axis="x", visible=False)
ax.set_xlabel("Team's buff exposure (within-year tercile)")
ax.set_ylabel("Opponent's excess bans per game\n(actual − expected)")
ax.legend(loc="upper right", frameon=False, fontsize=6.5, labelcolor=INK2)
panel_title(ax, "b", "Draft: excess bans by buff exposure")

ax = axes[3]
DE = pd.read_csv(OUT / "detectable_effect_summary.csv").set_index("scope")
for i, r in enumerate([DE.loc["game"], DE.loc["bo5"]]):
    ax.fill_between([-r.mde80_negative, r.mde80], i - 0.26, i + 0.26, color=GRID, lw=0, zorder=0.5)
    ax.annotate(f"80% power: −{r.mde80_negative:.1f}, +{r.mde80:.1f} pp", (r.mde80, i + 0.26), xytext=(0, -2), textcoords="offset points",
                ha="right", va="top", fontsize=6, color=MUTED)
    ax.plot([r.lo, r.hi], [i, i], color=S1, lw=1.5, solid_capstyle="round", zorder=2)
    ax.scatter(r.est, i, s=26, color=S1, edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(f"{r.est:+.1f} pp ({r.lo:+.1f} to {r.hi:+.1f})", (r.est, i), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=6.5, color=INK)
ax.axvline(0, color=AXIS, lw=1.1, zorder=1)
ax.set_yticks([0, 1], ["Per game", "Best-of-5\nseries"])
ax.set_ylim(1.7, -0.7)
ax.set_xlim(-10, 10)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.grid(axis="y", visible=False)
ax.set_xlabel("Estimated win-probability association\nper SD of net exposure (pp)")
panel_title(ax, "c", "Results: interval includes zero")
save(fig, "fig1_levels")

# ---------- Figure 2: mechanism ----------
C = pd.read_csv(OUT / "buff_persistence_champion_presence.csv")
C["decile"] = np.minimum((C.pres_pct * 10).apply(np.ceil).astype(int), 10)
dec = C.groupby("decile").agg(nerfed=("net", lambda s: (s < 0).mean() * 100), buffed=("net", lambda s: (s > 0).mean() * 100),
                              n=("net", "size"))
T = pd.read_csv(OUT / "buff_persistence_teams.csv")
T["qf"] = [(y, t) in KO8 for y, t in zip(T.year, T.teamid)]
T["champ"] = [(y, t) in CH.index for y, t in zip(T.year, T.teamid)]
T["reg"] = [(y, t) in REG8 for y, t in zip(T.year, T.teamid)]
assert T.reg.sum() == len(REG)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.32})
for col, colr, lab in [("nerfed", S2, "Nerfed"), ("buffed", S1, "Buffed")]:
    a1.plot(dec.index, dec[col], color=colr, lw=1.5, marker="o", ms=4, mec="white", mew=1, label=lab)
a1.set_xticks(range(1, 11))
a1.set_xlim(0.5, 10.5)
a1.set_ylim(0, None)
a1.set_xlabel("Pick-and-ban presence before Worlds, all recorded games\n(within-year decile; 10 = most contested)")
a1.set_ylabel("Champions changed in the patch window (%)")
a1.legend(loc="upper left", frameon=False, fontsize=6.5, labelcolor=INK2)
a1.grid(axis="x", visible=False)
panel_title(a1, "a", "Nerfs follow the contested meta")

oth = ~T.qf
a2.scatter(T.meta_z[oth], T.intended_z[oth], s=8, color=MUTED, alpha=0.45, edgecolor="white", linewidth=0.4, label="Other teams")
a2.scatter(T.meta_z[T.qf & ~T.champ], T.intended_z[T.qf & ~T.champ], s=10, color=QF, edgecolor="white", linewidth=0.4,
           label="Quarterfinalists")
a2.scatter(T.meta_z[T.champ], T.intended_z[T.champ], s=18, color=S1, edgecolor="white", linewidth=0.7, label="Worlds winner")
a2.scatter(T.meta_z[T.reg], T.intended_z[T.reg], s=30, facecolor="none", edgecolor=S2, linewidth=0.9, zorder=4,
           label="Regional winner")
fit = sm.OLS(T.intended_z, sm.add_constant(T.meta_z)).fit()
xs = np.linspace(T.meta_z.min(), T.meta_z.max(), 50)
a2.plot(xs, fit.params["const"] + fit.params["meta_z"] * xs, color=INK2, lw=1.2, zorder=1)
r = np.corrcoef(T.meta_z, T.intended_z)[0, 1]
a2.annotate(f"slope {fit.params['meta_z']:+.2f} SD per SD, r = {r:.2f}", (0.98, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=6.5, color=INK2)
t1 = T[(T.team == "T1") & T.year.between(2022, 2025)]
OFF = {2022: (4, 2), 2023: (5, 4), 2024: (4, 2), 2025: (5, -8)}
for _, rr in t1.iterrows():
    a2.annotate(f"T1 ’{str(rr.year)[2:]}", (rr.meta_z, rr.intended_z), xytext=OFF[rr.year], textcoords="offset points",
                fontsize=5.5, color=INK)
a2.axhline(0, color=AXIS, lw=1, zorder=0.5)
a2.axvline(0, color=AXIS, lw=1, zorder=0.5)
a2.set_xlabel("Meta concentration of pre-Worlds pool\n(within-year z)")
a2.set_ylabel("Net patch exposure (within-year z)")
a2.legend(loc="lower left", frameon=False, fontsize=6, labelcolor=INK2, handletextpad=0.1)
panel_title(a2, "b", "Meta concentration and net exposure")
fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.2)
save(fig, "fig2_mechanism")

# ---------- Figure A1: strength controls ----------
PM = pd.read_csv(OUT / "patch_models.csv").set_index(["model", "term"])
BR = pd.read_csv(OUT / "ban_pressure_result_models.csv").set_index("term")
SP = pd.read_csv(OUT / "stage_pooled_tests.csv").set_index("test").loc["S3 pick advantage -> win (controls pre-Worlds strength)"]
SS = pd.read_csv(OUT / "strength_sensitivity.csv").set_index(["row", "control"])
RG = pd.read_csv(OUT / "rating_validation_region_control.csv").set_index(["row", "control"])
main = {"Net exposure": tuple(v / 4 * 100 for v in PM.loc[("intended", "d_intended_z"), ["estimate", "boot_lo", "boot_hi"]]),
        "Buff exposure": tuple(v / 4 * 100 for v in PM.loc[("nerfs + buffs", "d_buffs_z"), ["estimate", "boot_lo", "boot_hi"]]),
        "Pick advantage": (SP.value / 4 * 100, SP.lo / 4 * 100, SP.hi / 4 * 100),
        "Excess bans": tuple(BR.loc["d_pull", ["winprob_pp", "boot_lo_pp", "boot_hi_pp"]])}
ctrls = [("1-year rating (main)", lambda k: main[k]),
         ("90-day season rating", lambda k: tuple(SS.loc[(k, "season"), ["winprob_pp", "lo", "hi"]])),
         ("1-year + region dummies", lambda k: tuple(RG.loc[(k, "Rating + region dummies"), ["winprob_pp", "lo", "hi"]])),
         ("No strength control", lambda k: tuple(SS.loc[(k, "none"), ["winprob_pp", "lo", "hi"]]))]
panels = [("Net exposure", "Net exposure (pre-event)"), ("Buff exposure", "Buff exposure (pre-event)"),
          ("Excess bans", "Excess bans (draft, game 1)"), ("Pick advantage", "Pick advantage (draft)")]
fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), sharey=True, sharex=True)
axes = axes.ravel()
for ax, (key, title) in zip(axes, panels):
    ax.axvline(0, color=AXIS, lw=1.1, zorder=1)
    for i, (lab, get) in enumerate(ctrls):
        est, lo, hi = get(key)
        c = S1 if i == 0 else MUTED
        ax.plot([lo, hi], [i, i], color=c, lw=1.5, solid_capstyle="round", zorder=2)
        ax.scatter(est, i, s=24, color="white" if i == 3 else c, edgecolor=c, linewidth=1.1, zorder=3)
        ax.annotate(f"{est:+.1f}", (hi, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5,
                    color=INK if i == 0 else INK2)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=6)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
for a in (axes[0], axes[2]):
    a.set_yticks(range(len(ctrls)), [c[0] for c in ctrls])
axes[0].set_ylim(len(ctrls) - 0.4, -0.6)
axes[0].set_xlim(-10, 8.5)
fig.supxlabel("Win-probability association per SD (pp, at a 50:50 game); lines = 95% event-bootstrap intervals",
              fontsize=7.5, color=INK2)
fig.tight_layout(w_pad=1.2, h_pad=1.2)
save(fig, "figA1_strength_controls")
print("saved", sorted(p.name for p in FIG.glob("*.pdf")))
