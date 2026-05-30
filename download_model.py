import os, shutil, time
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download

if os.path.exists("./models/multilingual"):
    shutil.rmtree("./models/multilingual")
    print("Cleared previous incomplete download")

for attempt in range(1, 6):
    try:
        print(f"Attempt {attempt}/5...")
        snapshot_download(
            repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_dir="./models/multilingual",
            local_dir_use_symlinks=False,
            max_workers=1,
            ignore_patterns=["*.onnx", "*.ot", "onnx/*"],  # skip optional files
        )
        print("Done!")
        break
    except Exception as e:
        print(f"Retrying... ({e})")
        time.sleep(5)