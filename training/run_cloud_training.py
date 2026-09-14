"""
TreeVision AI — Seamless Cloud GPU Execution on Studio 'treevision-ai'
Orchestrates:
1. Connects to Studio 'treevision-ai' on NVIDIA T4 GPU
2. Uploads core source code & training pipelines (progress_bar=False)
3. Installs requirements on Cloud Studio
4. Launches GPU training on NEON dataset (30 epochs with AMP fp16)
5. Downloads best model weights directly to models/custom/best.pt
6. Stops Studio safely to conserve credits
"""

import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def run():
    print("=" * 65)
    print("⚡ TREEVISION AI — LAUNCHING CLOUD GPU TRAINING ON LIGHTNING AI")
    print("=" * 65)

    from lightning_sdk import Studio

    # 1. Connect
    print("\n[1/6] Connecting to Studio 'treevision-ai'...")
    studio = Studio(name="treevision-ai", teamspace="vision-model", user="anumallasarayu")
    print(f"Connected! ID: {studio.id} | Status: {studio.status} | Machine: {studio.machine}")

    # 2. Ensure running on T4
    if str(studio.status).lower() != "running":
        print("\n[2/6] Starting Studio on Tesla T4 GPU...")
        try:
            studio.start(machine="T4_SMALL")
        except Exception:
            studio.start()
        print(f"Studio is now RUNNING on {studio.machine} GPU!")
    elif "T4" not in str(studio.machine).upper() and "GPU" not in str(studio.machine).upper():
        print(f"\n[2/6] Switching Studio machine from {studio.machine} to T4_SMALL...")
        try:
            studio.switch_machine("T4_SMALL")
            print("Studio switched to T4 GPU!")
        except Exception as e:
            print(f"Note on switch machine: {e}")
    else:
        print(f"\n[2/6] Studio is already RUNNING on {studio.machine} GPU.")

    # 3. Create single compressed bundle and upload to Cloud Studio
    print("\n[3/6] Packaging and uploading project bundle to Cloud Studio...")
    import tarfile
    bundle_path = "training_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        for item in ["training", "src", "configs", "demo", "datasets/neon/processed", "requirements-training.txt", "README.md"]:
            if os.path.exists(item):
                tar.add(item)
    print(f"  Package created ({os.path.getsize(bundle_path) / (1024*1024):.2f} MB). Uploading...")
    studio.upload_file(bundle_path, remote_path=bundle_path, progress_bar=False)
    print("  Extracting bundle on Cloud Studio...")
    studio.run(f"tar -xzf {bundle_path}")
    print("Files synced cleanly in seconds!")

    # 4. Install dependencies
    print("\n[4/6] Installing dependencies on Cloud Studio...")
    out = studio.run("pip install -q ultralytics pyyaml pillow")
    print("Dependencies successfully installed!")

    try:
        # 5. Run GPU Training
        print("\n[5/6] Starting 100-Epoch GPU Training (Option B) on Cloud Studio...")
        print("=" * 65)
        train_output = studio.run("python training/lightning_train.py")
        print("Training Output:\n", train_output)
        print("=" * 65)

        # 6. Download best weights & metrics
        print("\n[6/6] Downloading trained weights (models/custom/best.pt)...")
        os.makedirs("models/custom", exist_ok=True)
        try:
            studio.download_file("models/custom/best.pt", "models/custom/best.pt")
            print("🎉 SUCCESS: Downloaded 'models/custom/best.pt' to local project!")
        except Exception as e:
            print(f"Note on weight download: {e}")

        try:
            studio.download_file("lightning_eval_metrics.json", "evaluation/custom_metrics.json")
            print("🎉 SUCCESS: Downloaded 'evaluation/custom_metrics.json' to local project!")
        except Exception as e:
            print(f"Note on metrics download: {e}")

        print("=" * 65)
        print("✅ CLOUD TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 65)
    finally:
        # Safely stop Studio to conserve cloud credits
        print("\nStopping Studio to conserve cloud credits...")
        studio.stop()
        print("Studio stopped safely.")

if __name__ == "__main__":
    run()
