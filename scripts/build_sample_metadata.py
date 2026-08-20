"""
Build metadata/sample_metadata.csv for a small curated set of REAL images
you've placed under sample_data/, by looking each one up in the full
(private) metadata/cell_lines_drug_compounds.csv. This keeps sample_data/
perfectly consistent with the real dataset's labels/splits without
hand-editing a CSV.

Setup:
  1. Pick a small subset of real single-cell crops (see the README for a
     suggested size) and copy them into sample_data/, preserving the
     original `<setting>/<filename>` folder structure used by the real
     dataset, e.g.:
         sample_data/ONS76_C7_0.037uM/E07_T000_Z000_Cell0284.png
     (setting = the string in the `setting` column of the master CSV.)
  2. Run this script from the repo root:
         python scripts/build_sample_metadata.py

Any file under sample_data/ whose <setting>/<filename> isn't found in the
master CSV is skipped with a warning (e.g. a typo'd folder name).
"""
import csv
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
MASTER_CSV = BASE_DIR / "metadata" / "cell_lines_drug_compounds.csv"
SAMPLE_DIR = BASE_DIR / "sample_data"
OUT_CSV = BASE_DIR / "metadata" / "sample_metadata.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def main():
    if not MASTER_CSV.exists():
        raise SystemExit(f"Master metadata CSV not found at {MASTER_CSV}")
    if not SAMPLE_DIR.exists():
        raise SystemExit(
            f"{SAMPLE_DIR} does not exist. Create it and copy in a subset of real "
            f"images first (see the docstring at the top of this script)."
        )

    with open(MASTER_CSV, newline="") as f:
        by_relpath = {row["relpath"]: row for row in csv.DictReader(f)}
        fieldnames = list(next(iter(by_relpath.values())).keys()) if by_relpath else []

    matched, missing = [], []
    for path in sorted(SAMPLE_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relpath = f"{path.parent.name}/{path.name}"
        row = by_relpath.get(relpath)
        if row is None:
            missing.append(relpath)
        else:
            matched.append(row)

    if not matched:
        raise SystemExit(f"No files under {SAMPLE_DIR} matched the master CSV — nothing written.")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched)

    print(f"[INFO] Matched {len(matched)} images -> {OUT_CSV}")
    if missing:
        print(f"[WARN] {len(missing)} file(s) under {SAMPLE_DIR} were not found in {MASTER_CSV}:")
        for m in missing[:10]:
            print(f"         {m}")
        if len(missing) > 10:
            print(f"         ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
