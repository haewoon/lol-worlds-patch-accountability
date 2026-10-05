<!-- Code item/system effects 2024–2025 · launched 2026-09-23T15:03:41.680Z · Claude Code general-purpose subagent · model claude-opus-5-5 -->

You are coding which champions an item / rune / system / objective change in League of Legends affected, for a research project. Work from the patch-note text only (blind coding): do not look up esports results, which teams won, or what pro players picked.

## Input
Read data/patch_notes/systems_E.csv — a list of non-champion changes (columns year_group,patch_oe,category,name,change_summary,likely_affected) for patches 14.16–14.18 and client 15.18–15.20 (Riot titles "Patch 25.18/25.19/25.20"). The summaries are short: re-read the official patch notes (leagueoflegends.com game-updates, or https://wiki.leagueoflegends.com/en-us/ patch pages) for the details you need. Use WebSearch/WebFetch (load via ToolSearch if deferred).
Also read champion_names.txt in the same folder (exact champion spellings). Do not read other files in that folder.

## What to code
Skip rows that are not gameplay balance on Summoner's Rift (game modes such as ARAM/Arena/Doom Bots, cosmetics, client/honor/ranked, new-champion release notes, pure bug fixes with no balance effect). For every remaining change, write one or more rows saying WHO it made stronger or weaker:

year_group,patch_oe,name,target_type,target,direction,magnitude,confidence,note
- target_type:
  - subclass — target is one of: Enchanter, Catcher, Juggernaut, Diver, Burst, Battlemage, Artillery, Marksman, Assassin, Skirmisher, Vanguard, Warden, Specialist (Riot's official classes)
  - role — target is one of: top, jungle, mid, bot, support
  - champion — target is an exact name from champion_names.txt (use when the change mainly matters for specific champions, e.g. an item whose core users are well known in that era, or a rune a few champions rely on; write one row per champion)
  - team_style — target is one of: early_game, scaling, objective_control, teamfight, splitpush, vision (for objective/map/system changes that favour a way of playing rather than champions)
- direction: buff | nerf (for the target). If a change helps one group and hurts another, write separate rows.
- magnitude: 1 small, 2 moderate, 3 large (how much it plausibly moved that target's strength)
- confidence: high (the notes say who it targets, or it is mechanically obvious) | low (your inference)
- note: at most 15 words
Use patch_oe = client version (14.16 … 15.20). Prefer the narrowest target that is accurate (a champion list over a whole subclass when an item is used by a handful of champions). Do not code the same effect twice through overlapping targets.

## Output (overwrite if present)
data/patch_notes/sysfx_E.csv

## Report back (short)
Rows written per patch, how many input rows you skipped and why, and the changes you were least sure how to map.
