# Coding-agent prompts

Exact prompts given to the LLM coding agents (Claude Opus 5.5, `claude-opus-5-5`, run as Claude Code `general-purpose` subagents on 2026-09-23). Absolute scratch paths are shortened to `data/patch_notes/`.

Used in the paper: the five `code_patch_notes_*` prompts (coded_A–E) and `independent_re_code_for_reliability` (coded_R). The post-Worlds, item/system, and champion-class prompts belong to analyses not reported in the preprint.

Audit of the six champion coders' tool calls: web searches only for patch notes/hotfixes; hosts fetched only leagueoflegends.com (www, na), wiki.leagueoflegends.com, leagueoflegends.fandom.com, web.archive.org (no esports-results sites). The full tool-call lists of these six coders are in `tool_calls/` (one row per call: time, tool, and the query, URL, command, or file path; scratch paths and user names removed).
No champion coder opened the match data (Oracle's Elixir CSVs), any file in output/, or another coder's coded_*.csv (checked over all Bash/Read/Grep/Glob calls).

| launched | task | file |
|---|---|---|
| 2026-09-23 13:14 UTC | Code patch notes 2016–2017 | `code_patch_notes_2016_2017.md` |
| 2026-09-23 13:15 UTC | Code patch notes 2018–2019 | `code_patch_notes_2018_2019.md` |
| 2026-09-23 13:15 UTC | Code patch notes 2020–2021 | `code_patch_notes_2020_2021.md` |
| 2026-09-23 13:15 UTC | Code patch notes 2022–2023 | `code_patch_notes_2022_2023.md` |
| 2026-09-23 13:15 UTC | Code patch notes 2024–2025 | `code_patch_notes_2024_2025.md` |
| 2026-09-23 13:15 UTC | Independent re-code for reliability | `independent_re_code_for_reliability.md` |
| 2026-09-23 13:27 UTC | Code post-Worlds patches 2016–2017 | `code_post_worlds_patches_2016_2017.md` |
| 2026-09-23 13:27 UTC | Code post-Worlds patches 2018–2019 | `code_post_worlds_patches_2018_2019.md` |
| 2026-09-23 13:28 UTC | Code post-Worlds patches 2020–2021 | `code_post_worlds_patches_2020_2021.md` |
| 2026-09-23 13:28 UTC | Code post-Worlds patches 2022–2023 | `code_post_worlds_patches_2022_2023.md` |
| 2026-09-23 13:28 UTC | Code post-Worlds patches 2024–2025 | `code_post_worlds_patches_2024_2025.md` |
| 2026-09-23 15:02 UTC | Build champion class table | `build_champion_class_table.md` |
| 2026-09-23 15:03 UTC | Code item/system effects 2016–2017 | `code_item_system_effects_2016_2017.md` |
| 2026-09-23 15:03 UTC | Code item/system effects 2018–2019 | `code_item_system_effects_2018_2019.md` |
| 2026-09-23 15:03 UTC | Code item/system effects 2020–2021 | `code_item_system_effects_2020_2021.md` |
| 2026-09-23 15:03 UTC | Code item/system effects 2022–2023 | `code_item_system_effects_2022_2023.md` |
| 2026-09-23 15:03 UTC | Code item/system effects 2024–2025 | `code_item_system_effects_2024_2025.md` |
| 2026-09-23 15:03 UTC | Independent re-code of system effects | `independent_re_code_of_system_effects.md` |
