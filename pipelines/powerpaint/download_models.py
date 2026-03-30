from huggingface_hub import snapshot_download
from pathlib import Path

def download_models():
    pipeline_root = Path(__file__).resolve().parent

    # PowerPaint v1
    print("Downloading PowerPaint v1 model...")
    snapshot_download(
        repo_id="JunhaoZhuang/PowerPaint-v1",
        local_dir=str(pipeline_root / "checkpoints" / "ppt-v1"),
        local_dir_use_symlinks=False
    )
    print("PowerPaint v1 download complete!")

    # PowerPaint v2 (Optional - uncomment if you want v2 as well)
    # print("Downloading PowerPaint v2 model...")
    # snapshot_download(
        # repo_id="JunhaoZhuang/PowerPaint_v2",
        # local_dir="./checkpoints/ppt-v2",
        # local_dir_use_symlinks=False
    # )
    # print("PowerPaint v2 download complete!")

if __name__ == "__main__":
    download_models()
