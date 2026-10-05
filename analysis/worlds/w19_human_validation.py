"""W19: human validation of the LLM (subagent) patch-note coding.

  python w19_human_validation.py sample   writes paper/validation/human_coding_sheet.csv (12 champion changes per
                                          Worlds year, 2016-2025) and a separate answer key with the LLM codes.
                                          The coder reads the ORIGINAL patch note at source_url and fills
                                          human_direction (buff / nerf / adjust / rework / bugfix) and
                                          human_magnitude (1 small, 2 medium, 3 large; 0 for adjust/bugfix)
                                          without opening the key.
  python w19_human_validation.py score    agreement of human vs LLM codes (direction, sign kappa, magnitude).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from worlds_common import sr_only

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "paper" / "validation"
SHEET, KEY = VAL / "human_coding_sheet.csv", VAL / "llm_key_DO_NOT_OPEN_BEFORE_CODING.csv"
PER_YEAR = 12


def sample():
    if SHEET.exists():
        sys.exit(f"{SHEET} exists; not overwriting a sheet that may already be filled in")
    VAL.mkdir(parents=True, exist_ok=True)
    pre = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in sorted((ROOT / "data/patch_notes").glob("coded_[A-E].csv"))])
    pre = sr_only(pre)
    s = pre.groupby("year_group").sample(n=PER_YEAR, random_state=19)
    s = s.sample(frac=1, random_state=19).reset_index(drop=True)          # shuffle so years/patches are not in blocks
    s.insert(0, "id", range(1, len(s) + 1))
    s[["id", "year_group", "patch_oe", "riot_patch_title", "release_date", "hotfix", "champion", "source_url"]].assign(
        human_direction="", human_magnitude="", notes="").to_csv(SHEET, index=False)
    s[["id", "direction", "magnitude", "summary"]].to_csv(KEY, index=False)
    print(f"wrote {len(s)} rows -> {SHEET.relative_to(ROOT)}; key -> {KEY.relative_to(ROOT)}")


def score():
    h = pd.read_csv(SHEET, dtype={"patch_oe": str}).merge(pd.read_csv(KEY), on="id")
    h = h[h.human_direction.notna() & (h.human_direction.astype(str).str.strip() != "")]
    h["hd"] = h.human_direction.str.lower().str.strip()
    h["ld"] = h.direction.str.lower().str.strip()
    sign = lambda d: np.select([d == "buff", d == "nerf"], [1, -1], 0)
    hs, ls = sign(h.hd), sign(h.ld)
    po = np.mean(hs == ls)
    pe = sum(np.mean(hs == k) * np.mean(ls == k) for k in (-1, 0, 1))
    both = (hs != 0) & (hs == ls)
    print(f"coded rows: {len(h)}")
    print(f"direction exact agreement: {np.mean(h.hd == h.ld):.3f}")
    print(f"sign (buff / nerf / other) agreement: {po:.3f}, kappa {(po - pe) / (1 - pe):.3f}")
    if both.any():
        hm = pd.to_numeric(h.human_magnitude, errors="coerce")[both]
        print(f"magnitude, same-sign rows: exact {np.mean(hm == h.magnitude[both]):.3f}, "
              f"within 1 {np.mean((hm - h.magnitude[both]).abs() <= 1):.3f}, corr {np.corrcoef(hm, h.magnitude[both])[0, 1]:.3f}")
    print(pd.crosstab(h.hd, h.ld, rownames=["human"], colnames=["LLM"]).to_string())
    h.to_csv(VAL / "human_vs_llm.csv", index=False)


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
