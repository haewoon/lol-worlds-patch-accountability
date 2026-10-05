<!-- Code patch notes 2024–2025 · launched 2026-09-23T13:15:41.019Z · Claude Code general-purpose subagent · model claude-opus-5-5 -->

You are coding League of Legends patch notes for a research project. Work from the patch-note TEXT only (blind coding): do not use or look up esports results, which teams won, or what pro players picked. Your judgement must come only from what each patch changed.

## Patches to code (OE label = the client version number as "major.minor")
- Group year 2024: patches 14.16, 14.17, 14.18 (hotfix cutoff for 14.18: include only hotfixes released before 2024-09-25)
- Group year 2025: OE labels 15.18, 15.19, 15.20 (hotfix cutoff for 15.20: before 2025-10-14). In 2025 Riot titled patch notes by year (e.g. "Patch 25.19 Notes") while the client version stayed 15.x. Map OE 15.18/15.19/15.20 to the Riot patch-note titles by release date (the Worlds 2025 main event started 2025-10-14 on client version 15.20; the patches are two weeks apart) and record the Riot title you used in riot_patch_title.
For earlier patches in the list include all their hotfixes / mid-patch updates.

## Sources
Use the official Riot patch notes (leagueoflegends.com game-updates) and/or the League of Legends Wiki patch pages (https://leagueoflegends.fandom.com/wiki/V14.18 style; for 2025 check how the wiki names them) which list all changes incl. hotfixes. Use WebSearch/WebFetch (load them with ToolSearch if they are deferred). Prefer official notes; use the wiki to catch hotfixes and exact numbers.

## What to code
One row per champion per patch for every CHAMPION change (combine all lines for the same champion in the same patch into one net judgement). If a hotfix within that patch changes the champion again, add a separate row with hotfix=1.
Columns (CSV with header, comma-separated, quote the summary field):
year_group,patch_oe,riot_patch_title,release_date,champion,direction,magnitude,pro_targeted,hotfix,summary,source_url
- champion: must match EXACTLY one line of data/patch_notes/champion_names.txt (read it first; e.g. "Kai'Sa", "Nunu & Willump", "Wukong", "Dr. Mundo", "K'Sante", "Bel'Veth"). If a champion is missing from that list (e.g. a brand-new release), use its official name and flag it in your report.
- direction: buff (net power increase) | nerf (net power decrease) | adjust (mixed/sidegrade, net unclear) | rework (VGU / mini-rework, net unclear) | bugfix (only bug fixes, no intended balance change)
- magnitude (buff/nerf only; use 0 for adjust/rework/bugfix): 1 = small (one number, roughly <10% change to one ability/stat); 2 = moderate (several numbers or a meaningful change to a core ability/base stat — a typical headline buff/nerf); 3 = large (big multi-part change clearly meant to move the champion's viability, e.g. major pro-play nerf or large compensation buff)
- pro_targeted: 1 if the notes say the change targets pro/competitive/high-level/elite play, else 0
- summary: at most 20 words, what changed
Also write item / rune / summoner-spell / system / objective changes to a second CSV:
year_group,patch_oe,category,name,change_summary,likely_affected
(likely_affected = free text, e.g. "AD assassins", "tanks", "junglers", "early-game teams").

## Output files (write both, overwrite if present)
- data/patch_notes/coded_E.csv
- data/patch_notes/systems_E.csv
Do not read or modify any other file in data/patch_notes/ (other coders are working there independently).

## Report back (short)
Patches covered with champion-row counts each, the OE→Riot title mapping you used for 2025, any patch or hotfix you could not find, and a list of rows you were unsure about (champion, patch, why).
