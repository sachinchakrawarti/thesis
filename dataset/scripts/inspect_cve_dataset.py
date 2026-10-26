# scripts/inspect_cve_dataset.py
"""Download and inspect the SpongeBOB9684/CVE-TO-MITRE dataset."""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "dataset" / "cve_mitre"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Fetching dataset information from Hugging Face...")

try:
    from huggingface_hub import snapshot_download, dataset_info
    
    # 1. Check metadata (last modified, size)
    info = dataset_info("SpongeBOB9684/CVE-TO-MITRE")
    print(f"\n--- DATASET INFO ---")
    print(f"ID            : {info.id}")
    print(f"Last Modified : {info.last_modified}")
    print(f"Downloads/mo  : {info.downloads}")
    
    # 2. Download the dataset files
    print("\nDownloading dataset files (this may take a minute)...")
    path = snapshot_download(
        repo_id="SpongeBOB9684/CVE-TO-MITRE",
        repo_type="dataset",
        local_dir=OUT_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {path}")
    
    # 3. List files
    print("\n--- FILES DOWNLOADED ---")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(OUT_DIR)} ({size_kb:.1f} KB)")
    
    # 4. Inspect first few rows of each JSONL file
    print("\n--- SAMPLE ROWS ---")
    for jsonl_file in sorted(OUT_DIR.glob("**/*.jsonl")):
        print(f"\nFile: {jsonl_file.name}")
        with open(jsonl_file, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 2: break  # Show only first 2 rows per file
                row = json.loads(line)
                for key, val in row.items():
                    if isinstance(val, str) and len(val) > 120:
                        val = val[:120] + "..."
                    print(f"  {key}: {val}")
                print("  ---")
    
    # 5. Show README/citation if present
    readme = OUT_DIR / "README.md"
    if readme.exists():
        print("\n--- README (first 20 lines) ---")
        for i, line in enumerate(readme.read_text(encoding="utf-8").splitlines()):
            if i >= 20: break
            print(f"  {line}")

except Exception as e:
    print(f"\nError: {e}")
    print("\nTroubleshooting:")
    print("  - Make sure you have: pip install huggingface_hub")
    print("  - Check your internet connection.")
    print("  - If the dataset is gated, run: huggingface-cli login")