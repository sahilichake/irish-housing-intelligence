"""Download the full Property Price Register (PPR) CSV.

If the automatic download fails (the PPR site sometimes blocks scripts or has
certificate issues), download it manually:
  1. Open https://www.propertypriceregister.ie  -> "Download" -> "All (PPR-ALL.zip)"
  2. Put PPR-ALL.zip (or the extracted PPR-ALL.csv) into data/raw/
  3. Re-run:  python -m src.pipeline
"""
import io
import sys
import zipfile

import requests

from src.config import PPR_URL, RAW_CSV, RAW_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (irish-housing-intelligence; research project)"}


def extract_zip(zip_bytes: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        RAW_CSV.write_bytes(zf.read(csv_name))


def download_ppr(force: bool = False) -> bool:
    """Return True if data/raw/PPR-ALL.csv is available afterwards."""
    if RAW_CSV.exists() and not force:
        print(f"[download] Using existing {RAW_CSV}")
        return True

    manual_zip = RAW_DIR / "PPR-ALL.zip"
    if manual_zip.exists():
        print("[download] Found manually downloaded PPR-ALL.zip, extracting...")
        extract_zip(manual_zip.read_bytes())
        return True

    print("[download] Downloading Property Price Register (~25 MB)...")
    try:
        resp = requests.get(PPR_URL, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        extract_zip(resp.content)
        print(f"[download] Saved {RAW_CSV}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[download] Automatic download failed: {exc}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return False


if __name__ == "__main__":
    download_ppr(force="--force" in sys.argv)
