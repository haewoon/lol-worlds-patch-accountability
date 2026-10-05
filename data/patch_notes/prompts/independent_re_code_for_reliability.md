<!-- Independent re-code for reliability · launched 2026-09-23T13:15:52.492Z · Claude Code general-purpose subagent · model claude-opus-5-5 -->

You are an independent second coder of League of Legends patch notes for a research project (inter-coder reliability). Work from the patch-note TEXT only (blind coding): do not use or look up esports results, which teams won, or what pro players picked. Do not read any other file in the data/patch_notes/ folder except champion_names.txt — other coders are coding the same patches and your judgement must be independent.

## Patches to code (OE label = the client version number as "major.minor")
- Group year 2019: patches 9.16, 9.17, 9.18, 9.19 (hotfix cutoff for 9.19: include only hotfixes released before 2019-10-02)
- Group year 2022: patches 12.15, 12.16, 12.17, 12.18 (hotfix cutoff for 12.18: before 2022-09-29)
- Group year 2025: OE labels 15.18, 15.19, 15.20 (hotfix cutoff for 15.20: before 2025-10-14). In 2025 Riot titled patch notes by year (e.g. "Patch 25.19 Notes") while the client version stayed 15.x. Map OE 15.18/15.19/15.20 to Riot titles by release date (Worlds 2025 main event started 2025-10-14 on client version 15.20; patches are two weeks apart) and record the Riot title in riot_patch_title.
For earlier patches in each group include all their hotfixes / mid-patch updates.

## Sources
Official Riot patch notes (leagueoflegends.com game-updates) and/or the League of Legends Wiki patch pages (https://leagueoflegends.fandom.com/wiki/V9.19 style) which list all changes incl. hotfixes. Use WebSearch/WebFetch (load them with ToolSearch if they are deferred).

## What to code
One row per champion per patch for every CHAMPION change (combine all lines for the same champion in the same patch into one net judgement). If a hotfix within that patch changes the champion again, add a separate row with hotfix=1.
Columns (CSV with header, comma-separated, quote the summary field):
year_group,patch_oe,riot_patch_title,release_date,champion,direction,magnitude,pro_targeted,hotfix,summary,source_url
- champion: must match EXACTLY one line of data/patch_notes/champion_names.txt (read it first). If a champion is missing from that list, use its official name and flag it.
- direction: buff (net power increase) | nerf (net power decrease) | adjust (mixed/sidegrade, net unclear) | rework (VGU / mini-rework, net unclear) | bugfix (only bug fixes, no intended balance change)
- magnitude (buff/nerf only; use 0 for adjust/rework/bugfix): 1 = small (one number, roughly <10% change to one ability/stat); 2 = moderate (several numbers or a meaningful change to a core ability/base stat — a typical headline buff/nerf); 3 = large (big multi-part change clearly meant to move the champion's viability, e.g. major pro-play nerf or large compensation buff)
- pro_targeted: 1 if the notes say the change targets pro/competitive/high-level/elite play, else 0
- summary: at most 20 words

## Output file (overwrite if present)
- data/patch_notes/coded_R.csv
(No systems file needed.)

## Report back (short)
Patches covered with champion-row counts each, the OE→Riot mapping used for 2025, anything you could not find, and rows you were unsure about.
