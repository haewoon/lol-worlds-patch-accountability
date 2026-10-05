"""W25a: the list of mid-patch hotfixes used by the game-data check (w25).

Game-data snapshots reflect a patch at release, so a champion hotfixed during patch X-1 cannot be compared X-1 -> X.
This script builds the candidate list from two sources and writes data/patch_notes/hotfix_exclusions.csv:
  - League of Legends Wiki patch pages (https://wiki.leagueoflegends.com/en-us/V<patch>): the "Hotfixes" section only,
    read entry by entry. An entry whose subject is a champion name is a Summoner's Rift hotfix to that champion
    (mode "SR", excluded by w25). Entries for other modes ("Arena-specific balancing", "ARAM-specific balancing",
    "One for All ...") are recorded with that mode and the champions they list, but not excluded, because w25 compares
    Summoner's Rift values only. Items, runes and system entries are ignored.
  - Hotfix rows of the LLM coding (hotfix = 1; Summoner's Rift by construction), mode "SR (coded)".
The list was checked entry by entry against the wiki sections on 2026-09-29 (see `checked`); rerunning the script
reproduces it from the cached pages (data/gamedata_cache/wiki/, fetched on first run).
"""
import glob
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from worlds_common import ROOT

CACHE = Path(os.environ.get("GAMEDATA_CACHE", ROOT / "data" / "gamedata_cache")) / "wiki"
CACHE.mkdir(parents=True, exist_ok=True)
OTHER_MODE = re.compile(r"Arena|ARAM|One for All|URF|Swarm|Brawl|Nexus Blitz|Ultimate Spellbook", re.I)

names = json.loads(sorted(Path(ROOT / "data/gamedata_cache/ddragon").glob("champion_*.json"))[-1].read_text())["data"]
CHAMPS = {v["name"] for v in names.values()}
NAME_RE = re.compile(r"(?<![A-Za-z'])(" + "|".join(sorted(map(re.escape, CHAMPS), key=len, reverse=True)) + r")(?![A-Za-z'])")
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[’`]", "'", s)).strip()
riot_label = lambda p: f"{int(p.split('.')[0]) + 10}.{p.split('.')[1]}" if int(p.split(".")[0]) >= 15 else p

coded = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in sorted(glob.glob(str(ROOT / "data/patch_notes/coded_[A-E].csv")))])
key = lambda p: tuple(map(int, p.split(".")))
patches = sorted({p for p in coded.patch_oe.unique()} | {f"{p.split('.')[0]}.{int(p.split('.')[1]) - 1}"
                                                          for p in coded.patch_oe.unique()}, key=key)


def page(patch):
    path = CACHE / f"V{patch}.html"
    if not (path.exists() and path.stat().st_size):
        url = f"https://wiki.leagueoflegends.com/en-us/V{riot_label(patch)}"
        req = urllib.request.Request(url, headers={"User-Agent": "lol-worlds-patch-fairness research script"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")


rows = []
for patch in patches:
    soup = page(patch)
    url = f"https://wiki.leagueoflegends.com/en-us/V{riot_label(patch)}"
    for h2 in soup.find_all("h2"):
        if not re.search("hotfix|mid-patch", h2.get_text(" ", strip=True), re.I):
            continue
        wrap = h2.parent if "mw-heading2" in (h2.parent.get("class") or []) else h2
        date = ""
        for sib in wrap.find_next_siblings():                      # the section only: stop at the next h2 or footer
            cls = sib.get("class") or []
            if sib.name == "h2" or "mw-heading2" in cls or "navbox-wrapper" in cls or "hidden-metadata" in cls:
                break
            if sib.name == "h3" or "mw-heading3" in cls:
                date = norm(sib.get_text(" ", strip=True).replace("[ edit | edit source ]", ""))
            elif sib.name == "dl":
                subject = norm(sib.get_text(" ", strip=True))
                if subject in CHAMPS:
                    rows.append({"patch": patch, "date": date, "champion": subject, "mode": "SR", "source": url})
                elif OTHER_MODE.search(subject):
                    mode = OTHER_MODE.search(subject).group(0)
                    body = sib.find_next_sibling("ul")
                    for c in sorted(set(NAME_RE.findall(norm(body.get_text(" ", strip=True))))) if body else []:
                        rows.append({"patch": patch, "date": date, "champion": c, "mode": mode, "source": url})
for r in coded[coded.hotfix == 1].itertuples():
    rows.append({"patch": r.patch_oe, "date": str(r.release_date), "champion": r.champion, "mode": "SR (coded)",
                 "source": r.source_url})
H = pd.DataFrame(rows).drop_duplicates(["patch", "date", "champion", "mode"])
H["exclude"] = H["mode"].str.startswith("SR")
H["checked"] = "2026-09-29: entry read in the wiki Hotfixes section (or coded hotfix row)"
H = H.sort_values(["patch", "date", "champion"], key=lambda s: s.map(key) if s.name == "patch" else s)
H.to_csv(ROOT / "data/patch_notes/hotfix_exclusions.csv", index=False)
print(H.groupby(["patch", "mode"]).champion.apply(lambda s: ", ".join(sorted(set(s)))).to_string())
print(f"\n{int(H.exclude.sum())} Summoner's Rift hotfix entries; {int((~H.exclude).sum())} other-mode entries recorded")
