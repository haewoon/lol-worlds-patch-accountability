# When the Rule-Maker Runs the World Championship

Replication files for Haewoon Kwak, *When the Rule-Maker Runs the World Championship: Late Patches and Procedural Accountability in League of Legends* (preprint, 2026; arXiv link to be added).

The paper measures how the patches released shortly before each *League of Legends* World Championship (Worlds) from 2016 to 2025 fell on the participating teams. It combines Oracle's Elixir match data, games restored from Leaguepedia, and LLM coding of 1,090 champion changes across 40 patches. This repository contains the code, the coded patch notes, the validation data, every analysis output the paper cites, and the manuscript source.

## Contents

| Path | Contents |
|---|---|
| `analysis/download_oracles_elixir.py` | Downloads the Oracle's Elixir match files and checks them against the analyzed snapshot. |
| `analysis/worlds/` | Analysis scripts. `worlds_common.py` defines the data loader and output location; `run_revised_pipeline.py` runs the analyses in order; `verify_manuscript_numbers.py` recomputes the numbers in the manuscript from the outputs and compares them with `paper/main.tex`. |
| `data/manifests/` | Worlds dates and patches, regional winners, unavailable champions, team lineages, voided games, Oracle's Elixir file checksums, and the record checks and corrections applied to Oracle's Elixir. |
| `data/supplement/` | Games of Worlds participants restored from Leaguepedia, the agreement between the two sources, and the matching extracts used by the roster scan. |
| `data/patch_notes/` | Coded champion changes (`coded_A`–`coded_E`, independent re-code `coded_R`, post-Worlds patches `post_A`–`post_E`), the coders' prompts and tool-call lists (`prompts/`), and helper lists. |
| `paper/validation/` | The author's validation coding, the matching LLM codes, and a copy of what the coding page showed. |
| `output/worlds_revision_2026-09-24/` | Analysis outputs (see `output/README.md`). |
| `output/coding_sensitivity/` | Full reruns of the analyses under three alternative codings (Appendix B). |
| `paper/` | The manuscript (`main.pdf`), its source (`main.tex`, `refs.bib`, `main.bbl`), the figure script, and the figures. |

## Reproducing the analyses

The scripts need Python 3.12.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements-lock.txt` lists the full environment used for the paper.

1. **Download the match data.** Oracle's Elixir files are not redistributed here.

   ```bash
   .venv/bin/python analysis/download_oracles_elixir.py --years 2015-2025
   ```

   The script compares each file with the SHA-256 of the copy analyzed for the paper (downloaded on September 23, 2026). Oracle's Elixir occasionally revises past files, so a mismatch is reported, not treated as an error. Results from revised files can differ slightly from the paper.

2. **Run the analyses** (about 40 minutes). The outputs go to the folder given with `--out`.

   ```bash
   .venv/bin/python analysis/worlds/run_revised_pipeline.py --out output/worlds_revision_2026-09-24
   ```

3. **Rerun the coding sensitivity analyses** (about an hour; Appendix B, Table 5).

   ```bash
   .venv/bin/python analysis/worlds/w26_coding_sensitivity.py
   ```

4. **Draw the figures.**

   ```bash
   .venv/bin/python paper/figures.py
   ```

5. **Check the manuscript numbers.** The script exits with status 1 if any number in `paper/main.tex` does not match the outputs.

   ```bash
   .venv/bin/python analysis/worlds/verify_manuscript_numbers.py
   ```

6. **Build the manuscript.**

   ```bash
   cd paper && latexmk -pdf main.tex
   ```

### Preparation steps whose results are included

The following steps need network access and are not required to reproduce the analyses, because their results are in `data/` and `output/`. The pages and tables they download are not redistributed.

| Step | Script | Result in this repository |
|---|---|---|
| Restore games from Leaguepedia | `w0_leaguepedia_supplement.py fetch`, then `build` after `w0a` | `data/supplement/` |
| Check and correct Oracle's Elixir records | `w0a_oe_record_checks.py`, `w0c_oe_corrections.py` | `data/manifests/oe_record_checks.csv`, `oe_corrections.csv`, `oe_external_evidence.csv` |
| Score the author's validation coding | `w19_human_validation.py score` | `paper/validation/human_vs_llm.csv` |
| Compare codes with game data | `w25a_hotfix_list.py`, `w25_gamedata_direction.py` | `data/patch_notes/hotfix_exclusions.csv`, `output/.../gamedata_direction_*` |

The order is `w0 fetch`, `w0a`, `w0 build`, `w0c`, and then the analyses above.

## Patch-note coding

Claude Opus 5.5 agents coded the official patch notes on September 23, 2026, working from the patch-note text only. `data/patch_notes/prompts/` holds the exact prompts and, in `tool_calls/`, the full list of tool calls of the six coders used in the paper. Appendix B of the paper describes the coding scheme and the checks against the author's coding and against game data.

## Data sources and licenses

The code is released under the MIT License (`LICENSE`). The coded patch notes, validation data, analysis outputs, and manuscript are released under CC BY 4.0. Files derived from Leaguepedia are under CC BY-SA 3.0. `DATA_LICENSE.md` lists each source and its terms.

This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

## Contact

Haewoon Kwak, Indiana University Bloomington (hwkwak@iu.edu)
