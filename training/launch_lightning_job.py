"""
TreeVision AI — Automated Lightning AI Studio Orchestrator
Launches Studio 'treevision-ai' on T4 GPU, uploads codebase,
executes training, downloads best weights, and safely stops the Studio.
"""

import os
import sys
import time

def run_lightning_cloud_pipeline(machine_type="T4", epochs=30):
    print("=" * 65)
    print("⚡ TREEVISION AI — LIGHTNING AI CLOUD GPU ORCHESTRATOR")
    print(f"Target Studio: treevision-ai | Machine: {machine_type}")
    print("=" * 65)

    from lightning_sdk import Studio, Machine

    # 1. Connect to Studio
    print("\n[1/6] Connecting to Studio 'treevision-ai'...")
    studio = Studio(name="treevision-ai", teamspace="vision-model", user="anumallasarayu", create_ok=True)
    print(f"Connected! Studio ID: {studio.id} | Current Status: {studio.status}")

    # 2. Start Studio on GPU
    if str(studio.status).lower() != "running":
        print(f"\n[2/6] Starting Studio on {machine_type} GPU...")
        studio.start(machine=machine_type)
        print("Studio is now RUNNING on Cloud GPU!")
    else:
        print("\n[2/6] Studio is already RUNNING.")

    # 3. Upload project files
    print("\n[3/6] Uploading project files to Cloud Studio...")
    files_to_upload = [
        "app.py",
        "requirements-training.txt",
        "README.md"
    ]
    for f in files_to_upload:
        if os.path.exists(f):
            print(f"  Uploading {f}...")
            studio.upload_file(f, remote_path=f)

    folders_to_upload = [
        "src",
        "configs",
        "training",
        "demo"
    ]
    for folder in folders_to_upload:
        if os.path.exists(folder):
            print(f"  Uploading folder {folder}/...")
            studio.upload_folder(folder, remote_path=folder)

    print("Project upload complete!")

    # 4. Install training dependencies on Studio
    print("\n[4/6] Installing dependencies on Lightning Studio...")
    res = studio.run("pip install -r requirements-training.txt")
    print("Dependencies installed!")

    # 5. Run Training on Cloud GPU
    print(f"\n[5/6] Launching GPU Training (Epochs: {epochs})...")
    train_cmd = f"python training/lightning_train.py"
    res = studio.run(train_cmd)
    print("Cloud GPU Training completed!")

    # 6. Download best weights back to local machine
    print("\n[6/6] Downloading best trained model weights (models/custom/best.pt)...")
    os.makedirs("models/custom", exist_ok=True)
    try:
        studio.download_file("models/custom/best.pt", "models/custom/best.pt")
        print("🎉 Successfully downloaded 'models/custom/best.pt' to local project!")
    except Exception as e:
        print(f"Download note: {e}")

    # Ask or auto-stop
    print("\n" + "=" * 65)
    print("✅ LIGHTNING AI GPU RUN FINISHED SUCCESSFULLY!")
    print("Stopping Studio to conserve cloud credits...")
    studio.stop()
    print("Studio stopped safely.")
    print("=" * 65)

if __name__ == "__main__":
    run_lightning_cloud_pipeline(machine_type="T4", epochs=30)
