"""W25: text-independent check of the patch-note coding against game data.

For every coded (non-hotfix) champion change, compare the champion's numeric game values in that patch with
the previous patch:
  2016-2018 (6.x-8.x)   Riot Data Dragon championFull.json: base stats, spell cooldown / cost / range,
                        effect arrays, ratio vars
  2019 (9.x)            CommunityDragon champion .bin.json when the champion has spell value arrays in both
                        patches, otherwise Data Dragon (Riot moved spell values out of Data Dragon in 2019)
  2020-2025 (10.x-15.x) CommunityDragon: base stats, spell cooldown / mana / range / radius / ammo, named
                        DataValues, effect arrays, spell-calculation numbers (Data Dragon only if the file is missing)
Each changed value is signed by its magnitude, "larger is better", except names signalling cooldown, cost, delay,
cast time, recharge or a required count; tooltip-only values are ignored. Values whose rank-by-rank changes disagree in sign, or that appear or
disappear, count as reshaped. Per change: up-only, down-only, mixed, reshaped-only, or no numeric change.
Hotfix rows are excluded (game data are versioned per patch), as are changes whose patch or immediately
preceding patch has no snapshot (10.17 is missing, which also rules out 10.18), and changes to a champion that was
hotfixed on Summoner's Rift during the preceding patch: snapshots reflect a patch at release, so such a hotfix would be
attributed to the next patch (list: data/patch_notes/hotfix_exclusions.csv, from w25a). Only Summoner's Rift values
are compared; other-mode spells and values are recognised by name markers (see OTHER_SPELL / OTHER_VALUE). Raw files are cached locally, not redistributed.
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from worlds_common import OUT, ROOT, sr_only

CACHE = Path(os.environ.get("GAMEDATA_CACHE", ROOT / "data" / "gamedata_cache"))
DD, CD = "https://ddragon.leagueoflegends.com", "https://raw.communitydragon.org"
LOWER_BETTER = re.compile(r"cooldown|^cd|cd$|cost|^mana$|delay|casttime|windup|recharge|lockout|selfdamage|healthcost|"
                          r"attacksper|required", re.I)
TOOLTIP = re.compile(r"tooltip|^tt", re.I)
CD_STATS = {"baseHP", "hpPerLevel", "baseStaticHPRegen", "hpRegenPerLevel", "baseArmor", "armorPerLevel",
            "baseSpellBlock", "spellBlockPerLevel", "baseDamage", "damagePerLevel", "baseMoveSpeed", "attackRange",
            "attackSpeed", "attackSpeedPerLevel", "attackSpeedRatio"}
SPELL_LISTS = ("cooldownTime", "mana", "castRange", "castRadius", "mAmmoRechargeTime", "mMaxAmmo")
CALC_PART = re.compile(r"/(mCoefficient|mNumber|mLevel1Value|mStartValue|mEndValue|mValues)(?=/\d+$|$)")
# Summoner's Rift only: spell objects and values of other modes, identified by name markers in the game files
# (Odyssey, Doom Bots / NightmareBot, URF, Arena = "Cherry", Swarm = "Strawberry", ARAM), are ignored.
OTHER_SPELL = re.compile(r"Odyssey|NightmareBot|DoomBots|URF|Cherry|Strawberry")
OTHER_VALUE = re.compile(r"Cherry|ARAM|URF|Urf|Strawberry")
CD_RESOURCE = {"arBase", "arPerLevel", "arBaseStaticRegen", "arRegenPerLevel"}
for sub in ("ddragon", "cdragon"):
    (CACHE / sub).mkdir(parents=True, exist_ok=True)


def fetch(url, path):
    """Cached download; a '.missing' marker remembers a 404."""
    miss = path.with_suffix(path.suffix + ".missing")
    if path.exists() and path.stat().st_size:
        return path
    if miss.exists():
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lol-worlds-patch-fairness research script"})
        with urllib.request.urlopen(req, timeout=180) as r:
            path.write_bytes(r.read())
        return path
    except urllib.error.HTTPError as e:
        if e.code == 404:
            miss.touch()
        return None


# ---------- versions and names ----------
vpath = fetch(DD + "/api/versions.json", CACHE / "ddragon" / "versions.json")
versions = [v for v in json.loads(vpath.read_text()) if re.fullmatch(r"\d+\.\d+\.\d+", v)]
key = lambda s: tuple(map(int, s.split(".")))
mm = lambda v: ".".join(v.split(".")[:2])
dd_version = {}
for v in sorted(versions, key=key):
    dd_version.setdefault(mm(v), v)                      # earliest x.y.z of each patch
def prev_of(patch):
    """The immediately preceding patch in Riot's numbering; None when either snapshot is missing from Data Dragon
    (10.17 is absent, so 10.17 and 10.18 cannot be compared with their own predecessors)."""
    major, minor = map(int, patch.split("."))
    prev = f"{major}.{minor - 1}"
    return prev if minor > 1 and patch in dd_version and prev in dd_version else None


prev_patch = {p: prev_of(p) for p in dd_version}
latest = max(versions, key=key)
cjson = fetch(f"{DD}/cdn/{latest}/data/en_US/champion.json", CACHE / "ddragon" / f"champion_{latest}.json")
name2id = {v["name"]: k for k, v in json.loads(cjson.read_text())["data"].items()}

# ---------- coded rows ----------
pre = pd.concat([pd.read_csv(f, dtype={"patch_oe": str}) for f in sorted(glob.glob(str(ROOT / "data/patch_notes/coded_[A-E].csv")))])
pre = sr_only(pre)
pre["direction"] = pre.direction.str.lower().str.strip()
pre["magnitude"] = pd.to_numeric(pre.magnitude, errors="coerce").fillna(0)
n_hotfix = int((pre.hotfix == 1).sum())
rows = pre[pre.hotfix != 1].copy()
rows["cid"] = rows.champion.map(name2id)
assert rows.cid.notna().all(), rows[rows.cid.isna()].champion.unique()
rows["major"] = rows.patch_oe.str.split(".").str[0].astype(int)
rows["prev"] = rows.patch_oe.map(prev_patch)          # missing when a patch or its predecessor has no snapshot


# ---------- downloads ----------
def dd_file(patch):
    v = dd_version[patch]
    return fetch(f"{DD}/cdn/{v}/data/en_US/championFull.json", CACHE / "ddragon" / f"championFull_{v}.json")


def cd_file(patch, cid):
    c = cid.lower()
    return fetch(f"{CD}/{patch}/game/data/characters/{c}/{c}.bin.json", CACHE / "cdragon" / f"{patch}_{c}.json")


have = rows[rows.prev.notna()]
dd_needed = sorted({p for _, r in have.iterrows() for p in (r.patch_oe, r.prev) if r.major <= 9}, key=key)
cd_needed = sorted({(p, r.cid) for _, r in have.iterrows() for p in (r.patch_oe, r.prev) if r.major >= 9})
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(dd_file, dd_needed))
    list(ex.map(lambda t: cd_file(*t), cd_needed))
print(f"cached {len(dd_needed)} Data Dragon versions and {len(cd_needed)} CommunityDragon champion files in {CACHE}")


# ---------- value extraction ----------
@lru_cache(maxsize=None)
def dd_data(patch):
    f = dd_file(patch)
    return json.loads(f.read_text())["data"] if f else {}


def dd_values(patch, cid):
    c = dd_data(patch).get(cid)
    if c is None:
        return None
    vals = {f"stats:{k}": [v] for k, v in c["stats"].items() if isinstance(v, (int, float))}
    for i, s in enumerate(c["spells"]):
        k = "QWER"[i]
        for f in ("cooldown", "cost", "range"):
            if isinstance(s.get(f), list):
                vals[f"{k}:{f}"] = s[f]
        for j, e in enumerate(s.get("effect") or []):
            if isinstance(e, list):
                vals[f"{k}:effect{j}"] = e
        for var in s.get("vars") or []:
            co = var.get("coeff")
            vals[f"{k}:var{var.get('key')}"] = co if isinstance(co, list) else [co]
    return vals


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}/{i}")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


@lru_cache(maxsize=None)
def cd_values(patch, cid):
    f = cd_file(patch, cid)
    if not f:
        return None
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError:
        return None
    vals, n_dv, rk = {}, 0, {}
    # number of ranks per ability slot (Q, W, E, R) where the file records it; rank arrays hold a placeholder at
    # index 0 and values for ranks 1..n, the rest being padding. A spell is mapped to its slot through (1) its ability
    # folder, (2) the ability object that lists it (hashed keys), or (3) its script name (exact, then longest prefix).
    # Abilities of an alternate form (e.g. Jayce's cannon) take the basic-ability rank count when Q, W and E share
    # it. Without a record or a mapping, ranks 1-5 are compared.
    root = next((v for k, v in d.items() if k.endswith("CharacterRecords/Root") and isinstance(v, dict)), {})
    levels = [len(x.get("mRequirements") or []) for x in root.get("spellLevelUpInfo") or [] if isinstance(x, dict)]
    folder_slot, name_slot = {}, {}
    for i, sn in enumerate((root.get("spellNames") or [])[:4]):
        parts = sn.split("/")
        if len(parts) > 1:
            folder_slot[parts[0]] = i
        name_slot[parts[-1]] = i
    abilities = {k: v for k, v in d.items() if isinstance(v, dict) and v.get("__type") == "AbilityObject"}
    child_ability = {c: ab.get("mName") for ab in abilities.values() for c in (ab.get("mChildSpells") or []) + [ab.get("mRootSpell")]}
    basic = {levels[i] for i in range(min(3, len(levels)))}
    alt_rank = basic.pop() if len(basic) == 1 and len(levels) >= 3 else None
    alt_abilities = {ab.get("mName") for ab in abilities.values() if ab.get("mType") not in (2, 3)} - set(folder_slot)

    def ranks(key, obj):
        parts = key.split("/")
        ability = parts[-2] if len(parts) > 1 and parts[-2] != "Spells" else child_ability.get(key)
        slot = folder_slot.get(ability)
        if slot is None:
            names = [parts[-1], obj.get("mScriptName") or ""]
            slot = next((name_slot[n] for n in names if n in name_slot), None)
            if slot is None:
                pref = [(len(sn), i) for n in names for sn, i in name_slot.items() if n and n.startswith(sn)]
                slot = max(pref)[1] if pref else None
        if slot is not None and slot < len(levels) and 1 <= levels[slot] <= 6:
            return levels[slot]
        if slot is None and ability in alt_abilities and alt_rank and 1 <= alt_rank <= 6:
            return alt_rank
        return 5

    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        sp = v.get("mSpell")
        last = k.split("/")[-1]
        name = v.get("mScriptName") or last if last.startswith("{") else last      # hashed keys: use the script name
        if isinstance(sp, dict) and not OTHER_SPELL.search(name):
            before = set(vals)
            for dv in sp.get("DataValues") or sp.get("mDataValues") or []:
                if isinstance(dv, dict) and "mName" in dv and isinstance(dv.get("mValues"), list) \
                        and not OTHER_VALUE.search(dv["mName"]):
                    vals[f"{name}:{dv['mName']}"] = dv["mValues"]
                    n_dv += 1
            for i, e in enumerate(sp.get("mEffectAmount") or []):
                if isinstance(e, dict) and isinstance(e.get("value"), list):
                    vals[f"{name}:effect{i}"] = e["value"]
                    n_dv += 1
            for fld in SPELL_LISTS:
                if isinstance(sp.get(fld), list):
                    vals[f"{name}:{fld}"] = sp[fld]
            for fld in ("mCoefficient", "mCoefficient2", "missileSpeed"):
                if isinstance(sp.get(fld), (int, float)):
                    vals[f"{name}:{fld}"] = [sp[fld]]
            for cname, calc in (sp.get("mSpellCalculations") or {}).items():
                if OTHER_VALUE.search(cname):
                    continue
                for p, x in leaves(calc):
                    part = CALC_PART.search(p)
                    if part:                          # rank arrays (mValues/0..6) are kept together
                        vals.setdefault(f"{name}:calc{cname}{p[:part.end()]}", []).append(x)
            for key in set(vals) - before:
                rk[key] = ranks(k, v)
        if k.endswith("CharacterRecords/Root"):
            for fld, x in v.items():
                if fld in CD_STATS and isinstance(x, (int, float)):
                    vals[f"stats:{fld}"] = [x]
                if fld == "primaryAbilityResource" and isinstance(x, dict):
                    for g, y in x.items():
                        if g in CD_RESOURCE and isinstance(y, (int, float)):
                            vals[f"stats:resource.{g}"] = [y]
    return vals, n_dv, rk


def compare(A, B, cd=False, rk=None):
    """Signed moves between two value dicts: (ups, downs, reshaped, changed field list)."""
    ups = downs = reshaped = 0
    changed = []
    for k in sorted(set(A) | set(B)):
        field = k.split(":")[-1]
        if TOOLTIP.search(field):
            continue
        a, b = A.get(k), B.get(k)
        if a == b:
            continue
        if a is None or b is None or len(a) != len(b):
            reshaped += 1
            changed.append(f"{k}~")
            continue
        if cd and len(a) == 7:                # CommunityDragon rank arrays: index 0 is a placeholder, then ranks
            n = (rk or {}).get(k, 5)
            a, b = a[1:1 + n], b[1:1 + n]
        # compare magnitudes (slows and reductions are often stored as negative numbers); a sign flip is a reshape
        signs = {9 if x * y < 0 else int(np.sign(abs(y) - abs(x))) for x, y in zip(a, b)
                 if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x != y}
        if not signs:
            continue
        if len(signs) > 1 or 9 in signs:
            reshaped += 1
            changed.append(f"{k}~")
            continue
        s = signs.pop() * (-1 if LOWER_BETTER.search(field) else 1)
        ups += s > 0
        downs += s < 0
        changed.append(f"{k}{'+' if s > 0 else '-'}")
    return ups, downs, reshaped, changed


# ---------- hotfixes during the preceding patch ----------
# Snapshots reflect a patch at release, so a Summoner's Rift hotfix to patch X-1 first shows up in the X snapshot.
# Champions hotfixed during X-1 (data/patch_notes/hotfix_exclusions.csv, built and checked with w25a) are not compared.
HOT = pd.read_csv(ROOT / "data/patch_notes/hotfix_exclusions.csv", dtype={"patch": str})
sr_hotfix = set(zip(HOT.loc[HOT.exclude, "patch"], HOT.loc[HOT.exclude, "champion"]))
rows["prev_hotfix"] = [pd.notna(r.prev) and (r.prev, r.champion) in sr_hotfix for r in rows.itertuples()]
print(f"hotfix check: {int(rows.prev_hotfix.sum())} changes follow a Summoner's Rift hotfix to the same champion "
      "in the preceding patch")

# ---------- per coded change ----------
out = []
for _, r in rows.iterrows():
    src = None
    if pd.isna(r.prev) or r.prev_hotfix:
        pass
    elif r.major >= 9:
        a, b = cd_values(r.prev, r.cid), cd_values(r.patch_oe, r.cid)
        if a and b and (r.major >= 10 or (a[1] > 0 and b[1] > 0)):
            A, B, src, RK = a[0], b[0], "CommunityDragon", b[2]
    if src is None and pd.notna(r.prev) and not r.prev_hotfix:
        A, B = dd_values(r.prev, r.cid), dd_values(r.patch_oe, r.cid)
        src = "Data Dragon" if A is not None and B is not None else None
    if src is None:
        out.append({"year": r.year_group, "patch": r.patch_oe, "champion": r.champion, "llm_dir": r.direction,
                    "llm_mag": r.magnitude, "source": "unavailable",
                    "reason": "hotfix in preceding patch" if r.prev_hotfix else "no snapshot"})
        continue
    ups, downs, reshaped, changed = compare(A, B, cd=src == "CommunityDragon", rk=RK if src == "CommunityDragon" else None)
    data_dir = ("mixed" if ups and downs else "up" if ups else "down" if downs else "reshaped" if reshaped else "none")
    out.append({"year": r.year_group, "patch": r.patch_oe, "champion": r.champion, "llm_dir": r.direction,
                "llm_mag": r.magnitude, "source": src, "ups": ups, "downs": downs, "reshaped": reshaped,
                "data_dir": data_dir, "net": (ups - downs) / (ups + downs) if ups + downs else 0.0,
                "changed": "; ".join(changed[:12])})
G = pd.DataFrame(out)
G.to_csv(OUT / "gamedata_direction_rows.csv", index=False)

# ---------- agreement with the LLM coding ----------
G = G[G.source != "unavailable"].copy()
G["period"] = G.source + " " + np.where(G.year <= 2018, "2016-18", "2019-25")
tab = pd.crosstab(G.llm_dir, G.data_dir).reindex(columns=["up", "down", "mixed", "reshaped", "none"], fill_value=0)
tab.to_csv(OUT / "gamedata_direction_crosstab.csv")
print(f"\nrows compared {len(G)} (hotfix rows excluded: {n_hotfix}; unavailable: {len(out) - len(G)})")
print(tab.to_string())


def summary(df, dcol="llm_dir", mcol="llm_mag"):
    """Agreement of a coder's directions with the game data.
    One-directional rows: every value with a directional change moved the same way (values that were reshaped may
    also be present; `strict` excludes them). The label-independent rate counts every one-directional row, so a
    buff/nerf coded as adjustment, rework or bug fix is a mismatch; the conditional rate keeps only rows the coder
    coded as buff or nerf. Rates on the same rows are comparable across coders."""
    code = np.select([df[dcol] == "buff", df[dcol] == "nerf"], [1, -1], 0)
    data = np.select([df.data_dir == "up", df.data_dir == "down"], [1, -1], 0)
    one = data != 0
    strict = one & (df.reshaped.fillna(0) == 0)
    signed = one & (code != 0)
    rate = lambda m: float((code[m] == data[m]).mean()) if m.any() else None
    return {"n": len(df),
            "n_one_directional": int(one.sum()), "sign_match_one_directional": rate(one),
            "n_one_directional_strict": int(strict.sum()), "sign_match_one_directional_strict": rate(strict),
            "n_one_directional_coded_signed": int(signed.sum()), "sign_match_one_directional_coded_signed": rate(signed),
            "buff_up_only": float((df[df[dcol] == "buff"].data_dir == "up").mean()),
            "buff_down_only": float((df[df[dcol] == "buff"].data_dir == "down").mean()),
            "nerf_down_only": float((df[df[dcol] == "nerf"].data_dir == "down").mean()),
            "nerf_up_only": float((df[df[dcol] == "nerf"].data_dir == "up").mean()),
            "bugfix_no_numeric_change": float((df[df[dcol] == "bugfix"].data_dir == "none").mean()),
            "data_up_coded_buff": float((df[df.data_dir == "up"][dcol] == "buff").mean()),
            "data_down_coded_nerf": float((df[df.data_dir == "down"][dcol] == "nerf").mean()),
            "data_none_coded_bugfix": float((df[df.data_dir == "none"][dcol] == "bugfix").mean()),
            "corr_signed_score_net": float(np.corrcoef(code * df[mcol].fillna(0), df.net)[0, 1])}


S = {"all": summary(G), **{p: summary(g) for p, g in G.groupby("period")}}
# the author's 120-row sample (non-hotfix rows)
hv = pd.read_csv(ROOT / "paper/validation/human_vs_llm.csv", dtype={"patch_oe": str})
hv = hv[hv.hotfix != 1].merge(G, left_on=["patch_oe", "champion"], right_on=["patch", "champion"], how="inner")
S["author_sample_llm"] = summary(hv)
S["author_sample_author"] = summary(hv, "hd", "human_magnitude")
dis = hv[hv.hd != hv.ld]
S["author_llm_disagreements"] = dis[["patch", "champion", "hd", "ld", "data_dir", "ups", "downs", "changed"]].to_dict("records")
(OUT / "gamedata_direction_summary.json").write_text(json.dumps(S, indent=2, default=str) + "\n")
print(json.dumps({k: v for k, v in S.items() if k != "author_llm_disagreements"}, indent=1))
print(pd.DataFrame(S["author_llm_disagreements"]).drop(columns="changed").to_string(index=False))
