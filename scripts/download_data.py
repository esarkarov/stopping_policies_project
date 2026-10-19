import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.hotpot_convert import parquet_to_official_json 

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOT_FILE = "hotpot_dev_distractor_v1.json"
HOTPOT_HF_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
MUSIQUE_DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
MUSIQUE_ZIP = "musique_data_v1.0.zip"
MUSIQUE_FILE = "musique_ans_v1.0_dev.jsonl"


def fetch(url: str, target: Path, timeout: int = 30) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out)


def download_hotpotqa(raw_dir: Path, force: bool) -> None:
    target = raw_dir / HOTPOT_FILE
    if target.exists() and not force:
        print(f"HotpotQA: {target} already exists, skipping")
        return
    try:
        print(f"HotpotQA: downloading {HOTPOT_URL}")
        fetch(HOTPOT_URL, target)
    except OSError as error:
        print(f"HotpotQA: official server unavailable ({error}); using the Hugging Face copy")
        parquet = raw_dir / "hotpot_dev_distractor.parquet"
        fetch(HOTPOT_HF_URL, parquet, timeout=120)
        count = parquet_to_official_json(parquet, target)
        print(f"HotpotQA: converted {count} questions to the official JSON format")
    print(f"HotpotQA: saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def download_musique(raw_dir: Path, force: bool) -> None:
    target = raw_dir / MUSIQUE_FILE
    if target.exists() and not force:
        print(f"MuSiQue: {target} already exists, skipping")
        return
    archive = raw_dir / MUSIQUE_ZIP
    if not archive.exists() or force:
        import gdown

        print("MuSiQue: downloading from Google Drive")
        result = gdown.download(id=MUSIQUE_DRIVE_ID, output=str(archive), quiet=False)
        if result is None:
            raise SystemExit(
                "MuSiQue download failed (Google Drive limit or no network).\n"
                f"Download the file manually from\n"
                f"https://drive.google.com/file/d/{MUSIQUE_DRIVE_ID}/view\n"
                f"save it as {archive} and run this script again."
            )
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if Path(m).name == MUSIQUE_FILE]
        if not members:
            raise SystemExit(f"{MUSIQUE_FILE} not found inside {archive}")
        with zf.open(members[0]) as src, open(target, "wb") as dst:
            dst.write(src.read())
    print(f"MuSiQue: saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="download again")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    raw_dir = ROOT / cfg["paths"]["data_dir"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if "hotpotqa" in cfg["data"]["datasets"]:
        download_hotpotqa(raw_dir, args.force)
    if "musique" in cfg["data"]["datasets"]:
        download_musique(raw_dir, args.force)


if __name__ == "__main__":
    main()