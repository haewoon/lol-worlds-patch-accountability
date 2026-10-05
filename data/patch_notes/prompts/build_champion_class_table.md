<!-- Build champion class table · launched 2026-09-23T15:02:50.441Z · Claude Code general-purpose subagent · model claude-opus-5-5 -->

Build a champion classification table for a League of Legends research project.

Read the champion list at data/patch_notes/champion_names.txt (173 names; use these exact spellings). Add any champion that exists in the game but is missing from that list (e.g. very new releases) as extra rows with a flag.

For every champion give:
- primary_subclass and secondary_subclass (secondary may be empty) from Riot's official class system, using exactly these 13 labels: Enchanter, Catcher, Juggernaut, Diver, Burst, Battlemage, Artillery, Marksman, Assassin, Skirmisher, Vanguard, Warden, Specialist. Source: the official League of Legends Wiki "List of champions" / champion pages (https://wiki.leagueoflegends.com/en-us/ — the fandom wiki may block automated access). If a champion's class changed over time, use the current one and note it.
- ddragon_tags: the Data Dragon "tags" field joined with "|" (e.g. "Fighter|Tank"), from https://ddragon.leagueoflegends.com/api/versions.json (take the newest version) then https://ddragon.leagueoflegends.com/cdn/<version>/data/en_US/champion.json. Map Data Dragon names to the list spellings (e.g. MonkeyKing = Wukong, "Nunu & Willump").
- damage_profile: one of AD, AP, mixed (your best judgement from the official wiki's champion info if it lists it; otherwise from the kit).
- crit_or_onhit (marksmen/skirmishers only, else empty): crit | onhit | lethality | other.

Use WebFetch/WebSearch/Bash curl (load via ToolSearch if deferred).

Write CSV (header row; quote fields that contain commas) to:
data/patch_notes/champion_classes.csv
columns: champion,primary_subclass,secondary_subclass,ddragon_tags,damage_profile,crit_or_onhit,note
Do not read or modify any other file in data/patch_notes/ except champion_names.txt.

Report back briefly: row count, champions you could not classify from the wiki (and what you used instead), and any name mismatches.
