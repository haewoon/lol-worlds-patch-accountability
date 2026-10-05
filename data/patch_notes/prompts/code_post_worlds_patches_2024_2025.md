<!-- Code post-Worlds patches 2024–2025 · launched 2026-09-23T13:28:28.490Z · Claude Code general-purpose subagent · model claude-opus-5-5 -->

You are coding League of Legends patch notes for a research project. Work from the patch-note TEXT only (blind coding): do not look up esports results, which teams won, or what pro players picked.

## Patches to code (OE label = client version "major.minor")
- Group year 2024: the five patches after 14.18 → 14.19, 14.20, 14.21, 14.22, 14.23
- Group year 2025: the five patches after client version 15.20 → 15.21, 15.22, 15.23, 15.24, and the first 2026 patch (client 16.1). In 2025–26 Riot titles patch notes by year (e.g. "Patch 25.21 Notes", "Patch 26.1 Notes") while the client version is 15.x/16.x; map by release date (patches ~two weeks apart; client 15.20 was live on 2025-10-14) and record the Riot title in riot_patch_title and the client version in patch_oe.
Include each patch's hotfixes / mid-patch updates.

## Sources
Official Riot patch notes (leagueoflegends.com game-updates) and/or League of Legends Wiki patch pages. Use WebSearch/WebFetch (load via ToolSearch if deferred).

## What to code
One row per champion per patch for every CHAMPION balance change (combine lines for the same champion in the same patch into one net judgement; a later hotfix in the same patch gets its own row with hotfix=1). Skip game-mode-only (ARAM/URF/OFA/Arena), visual, VO and tooltip-only changes.
CSV columns (header, comma-separated, quote text fields):
year_group,patch_oe,riot_patch_title,release_date,champion,direction,magnitude,pro_targeted,hotfix,reason,refers_to_patch,summary,source_url
- champion: must match EXACTLY one line of data/patch_notes/champion_names.txt (read it first). If a champion is missing from the list use its official name and flag it. New champion releases: no row (note them in your report).
- direction: buff | nerf | adjust (mixed, net unclear) | rework (VGU/mini-rework, net unclear) | bugfix
- magnitude (buff/nerf only, else 0): 1 small (one number, ~<10% to one ability/stat); 2 moderate (several numbers or a meaningful core change — typical headline change); 3 large (big multi-part change clearly meant to move viability)
- pro_targeted: 1 if the champion's own notes say the change targets pro/competitive/elite/high-skill play, else 0
- reason — the stated or evident reason, one of:
  pro_play (notes cite pro/competitive/tournament play or presence) |
  general (notes cite overall/solo-queue performance or no pro mention) |
  system_compensation (change exists to compensate for item/rune/system/preseason/season-start changes) |
  revert (explicitly reverts or walks back an earlier change) |
  unstated (no reason given)
- refers_to_patch: if the notes explicitly reference an earlier change, put that patch as the client version major.minor (e.g. 14.18 or 15.20); else empty
- summary: at most 20 words

## Output file (overwrite if present)
data/patch_notes/post_E.csv
Do not read or modify any other file in data/patch_notes/ except champion_names.txt.

## Report back (short)
Row counts per patch, the client→Riot title mapping, anything you could not find, which patch was the preseason/season-start patch, and rows you were unsure about.
