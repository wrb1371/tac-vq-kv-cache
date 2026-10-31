"""Download pinned public model and complete video benchmark; no scoring."""

from pathlib import Path
import concurrent.futures, hashlib, json, os, time, zipfile

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def model():
    repo = "Qwen/Qwen2.5-VL-7B-Instruct"
    rev = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    dest = ROOT / "artifacts/ar26/models/Qwen2.5-VL-7B-Instruct"
    snapshot_download(repo, revision=rev, local_dir=dest, max_workers=5)
    (ROOT / "data/ar26/MODEL_DOWNLOAD.json").write_text(
        json.dumps(
            {
                "completed": True,
                "repo": repo,
                "revision": rev,
                "local_directory": str(dest),
                "files": {
                    p.name: p.stat().st_size for p in dest.iterdir() if p.is_file()
                },
                "epoch": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    print("MODEL_DOWNLOAD_COMPLETE", flush=True)


def dataset():
    repo = "lmms-eval/TempCompass"
    rev = HfApi().dataset_info(repo).sha
    dest = ROOT / "datasets/ar26/TempCompass"
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=rev,
        local_dir=dest,
        allow_patterns=["README.md", "multi-choice/*", "tempcompass_videos.zip"],
        max_workers=3,
    )
    z = dest / "tempcompass_videos.zip"
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    with zipfile.ZipFile(z) as f:
        for name in f.namelist():
            p = (dest / name).resolve()
            assert p.is_relative_to(dest.resolve()), name
        f.extractall(dest)
    (ROOT / "data/ar26/DATASET_DOWNLOAD.json").write_text(
        json.dumps(
            {
                "completed": True,
                "repo": repo,
                "revision": rev,
                "local_directory": str(dest),
                "ZIP_SHA256": digest,
                "ZIP_bytes": z.stat().st_size,
                "epoch": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    print("DATASET_DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(model), pool.submit(dataset)]
        for f in futures:
            f.result()
