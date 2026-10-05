# Guide shown on the author's coding page

The author coded the 120 sampled changes on a single-page tool. The page showed the guide below (verbatim) and,
for each row, the information saved in `coding_page_rows.json`. The LLM codes were not on the page.

**Direction**

| Code | Meaning |
|---|---|
| Buff | Net power increase |
| Nerf | Net power decrease |
| Adjust | Mixed changes or a sidegrade; net unclear |
| Rework | Visual or gameplay update; net effect unclear |
| Bug fix | Only bug fixes, no intended balance change |

**Magnitude (buff / nerf only)**

| Code | Meaning |
|---|---|
| 1 | Small: about one number, under ~10% of one ability or stat |
| 2 | Moderate: several numbers, or a meaningful change to a core ability or base stat |
| 3 | Large: a multi-part change clearly meant to move the champion's viability |
| 0 | Set automatically for Adjust, Rework and Bug fix |

- One net judgement per champion per patch
- Hotfix rows: code the hotfix change only
- Ignore ARAM / Arena
- Written lines aren't reproduced; open the notes when a row says so
