# Human validation of the patch-note codes

| File | Contents |
|---|---|
| `coding_page_guide.md` | The coding scheme as shown on the author's coding page (Appendix B, "Coding scheme"). |
| `coding_page_rows.json` | The 120 sampled changes as shown on the page: patch, champion, release date, hotfix flag, links to the official notes and the wiki page, and the numeric lines extracted from the notes (`shown`). No LLM code is included. |
| `human_coding_sheet.csv` | The author's first-pass codes, entered without access to the LLM codes. The `notes` column keeps the author's working notes as written, in Korean. |
| `llm_key_DO_NOT_OPEN_BEFORE_CODING.csv` | The LLM codes for the same rows, kept separate until the author had finished. |
| `human_vs_llm.csv` | The two codings side by side, as scored by `analysis/worlds/w19_human_validation.py score`. |
