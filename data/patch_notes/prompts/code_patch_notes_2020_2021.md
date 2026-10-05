<!-- Code patch notes 2020–2021 · launched 2026-09-23T13:15:16.064Z · Claude Code general-purpose subagent · model claude-opus-5-5 -->

You are coding League of Legends patch notes for a research project. Work from the patch-note TEXT only (blind coding): do not use or look up esports results, which teams won, or what pro players picked. Your judgement must come only from what each patch changed.

## Patches to code (OE label = the version number as "major.minor")
- Group year 2020: patches 10.17, 10.18, 10.19 (hotfix cutoff for 10.19: include only hotfixes released before 2020-09-25)
- Group year 2021: patches 11.16, 11.17, 11.18, 11.19 (hotfix cutoff for 11.19: before 2021-10-05)
For earlier patches in the list include all their hotfixes / mid-patch updates.

## Sources
Use the official Riot patch notes (leagueoflegends.com game-updates "Patch 10.19 notes") and/or the League of Legends Wiki patch pages (https://leagueoflegends.fandom.com/wiki/V10.19 style) which list all changes incl. hotfixes. Use WebSearch/WebFetch (load them with ToolSearch if they are deferred). Prefer official notes; use the wiki to catch hotfixes and exact numbers.

## What to code
One row per champion per patch for every CHAMPION change (combine all lines for the same champion in the same patch into one net judgement). If a hotfix within that patch changes the champion again, add a separate row with hotfix=1.
Columns (CSV with header, comma-separated, quote the summary field):
year_group,patch_oe,riot_patch_title,release_date,champion,direction,magnitude,pro_targeted,hotfix,summary,source_url
- champion: must match EXACTLY one line of data/patch_notes/champion_names.txt (read it first; e.g. "Kai'Sa", "Nunu & Willump", "Wukong", "Dr. Mundo").
- direction: buff (net power increase) | nerf (net power decrease) | adjust (mixed/sidegrade, net unclear) | rework (VGU / mini-rework, net unclear) | bugfix (only bug fixes, no intended balance change)
- magnitude (buff/nerf only; use 0 for adjust/rework/bugfix): 1 = small (one number, roughly <10% change to one ability/stat); 2 = moderate (several numbers or a meaningful change to a core ability/base stat — a typical headline buff/nerf); 3 = large (big multi-part change clearly meant to move the champion's viability, e.g. major pro-play nerf or large compensation buff)
- pro_targeted: 1 if the notes say the change targets pro/competitive/high-level/elite play, else 0
- summary: at most 20 words, what changed
Also write item / rune / summoner-spell / system / objective changes to a second CSV:
year_group,patch_oe,category,name,change_summary,likely_affected
(likely_affected = free text, e.g. "AD assassins", "tanks", "junglers", "early-game teams").

## Output files (write both, overwrite if present)
- data/patch_notes/coded_C.csv
- data/patch_notes/systems_C.csv
Do not read or modify any other file in data/patch_notes/ (other coders are working there independently).

## Report back (short)
Patches covered with champion-row counts each, any patch or hotfix you could not find, and a list of rows you were unsure about (champion, patch, why).
