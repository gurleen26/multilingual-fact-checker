import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download
import time

for attempt in range(1, 6):
    try:
        print(f"Attempt {attempt}/5...")
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir="./models/all-MiniLM-L6-v2",
            local_dir_use_symlinks=False,
            max_workers=1,        # one file at a time — more stable
        )
        print("Done!")
        break
    except Exception as e:
        print(f"Timed out. Retrying in 5 seconds... ({e})")
        time.sleep(5)