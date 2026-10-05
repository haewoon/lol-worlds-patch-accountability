"""Download the Oracle's Elixir match files into data/ and check them against the analysed snapshot.

Oracle's Elixir (https://oracleselixir.com/tools/downloads) publishes one CSV per year in a public Google Drive
folder. File IDs are looked up by name in that folder (Drive IDs change when a file is re-uploaded) and fall back
to the IDs recorded in data/manifests/oracles_elixir_files.csv. Each file is then compared with the SHA-256 of
the copy analysed for the manuscript (downloaded 2026-09-23). The current-year file is updated daily, and past
files are occasionally revised, so a mismatch is reported rather than treated as an error.

  python analysis/download_oracles_elixir.py                  download missing files, verify all
  python analysis/download_oracles_elixir.py --years 2015-2025
  python analysis/download_oracles_elixir.py --write-manifest  record the local files as the snapshot
"""
import argparse
import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifests" / "oracles_elixir_files.csv"
FOLDER = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
NAME = "{}_LoL_esports_match_data_from_OraclesElixir.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def folder_ids():
    import gdown
    try:
        files = gdown.download_folder(id=FOLDER, skip_download=True, quiet=True)
        return {Path(f.path).name: f.id for f in files}
    except Exception as e:                     # listing can fail when Drive rate-limits; recorded IDs remain
        print(f"folder listing failed ({e}); using recorded file IDs")
        return {}


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--years", help="e.g. 2015-2025 (default: all years in the manifest)")
parser.add_argument("--write-manifest", action="store_true")
args = parser.parse_args()

if args.write_manifest:
    ids = folder_ids()
    rows = []
    for f in sorted(p for p in DATA.glob(NAME.format("*")) if not p.name.startswith("._")):
        year = int(f.name[:4])
        rows.append({"year": year, "file": f.name, "drive_id": ids.get(f.name, ""), "bytes": f.stat().st_size,
                     "sha256": sha256(f), "snapshot": "2026-09-23",
                     "note": "updated daily by Oracle's Elixir; the manuscript uses games before the 2025 Worlds"
                             if year == 2026 else ""})
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print(f"wrote {MANIFEST} ({len(rows)} files)")
    raise SystemExit

M = pd.read_csv(MANIFEST)
if args.years:
    lo, hi = (int(y) for y in args.years.split("-"))
    M = M[M.year.between(lo, hi)]
missing = M[[not (DATA / f).exists() for f in M.file]]
if len(missing):
    import gdown
    ids = folder_ids()
    for r in missing.itertuples():
        print(f"downloading {r.file}")
        gdown.download(id=ids.get(r.file, r.drive_id), output=str(DATA / r.file), quiet=True)
for r in M.itertuples():
    path = DATA / r.file
    if not path.exists():
        print(f"MISSING   {r.file}")
    elif sha256(path) == r.sha256:
        print(f"ok        {r.file}")
    else:
        print(f"DIFFERENT {r.file}: not the {r.snapshot} snapshot"
              + (" (expected: this file is updated daily)" if isinstance(r.note, str) and "daily" in r.note else
                 " (Oracle's Elixir revised it; results may differ slightly)"))
print(f"checked {len(M)} files on {dt.date.today()}")
