import os
import sys
import yaml
import shutil
import random
import boto3
from pathlib import Path
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import config (assuming similar structure to API config or standalone)
# For simplicity, we'll try to load from a yaml or env, but here we hardcode or use explicit args
# Ideally, reuse the same config object used in API.
try:
    from config import settings
except ImportError:
    # Fallback/Dummy settings if not running in same env as API
    class Settings:
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
        S3_BUCKET_NAME = "final-project-podo" # Replace with actual if needed
        DATASET_DIR = os.path.join(project_root, "PCB_DATASET")
    settings = Settings()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

def sync_refined_data(local_dataset_msg_dir: str):
    """
    Downloads new files from S3 refined/images and refined/labels
    Returns a list of new image filenames (stem).
    """
    bucket = settings.S3_BUCKET_NAME
    prefix = "refined/"
    
    # Ensure directories exist
    images_dir = Path(local_dataset_msg_dir) / "images"
    labels_dir = Path(local_dataset_msg_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Syncing from s3://{bucket}/{prefix}...")
    
    # List all objects in refined/
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    new_files = []
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            
            # Skip directories
            if not filename: 
                continue

            # Check if it's image or label
            if key.startswith("refined/images/") and (filename.endswith('.jpg') or filename.endswith('.png')):
                target_path = images_dir / filename
                if not target_path.exists(): # Only download if not exists
                    s3_client.download_file(bucket, key, str(target_path))
                    new_files.append(target_path.stem)
                    # Also try to download corresponding label
                    label_key = key.replace("images/", "labels/").replace(Path(key).suffix, ".txt")
                    label_path = labels_dir / (Path(filename).stem + ".txt")
                    try:
                        s3_client.download_file(bucket, label_key, str(label_path))
                    except Exception:
                        print(f"⚠️ Warning: Missing label for {filename}")
            
    print(f"✅ Downloaded {len(new_files)} new images.")
    return new_files

def update_split_files(dataset_dir, new_stems):
    """
    Appends new files to train.txt and val.txt (9:1 split).
    """
    if not new_stems:
        print("ℹ️  No new files to add.")
        return

    random.shuffle(new_stems)
    
    # 9:1 Split
    split_idx = int(len(new_stems) * 0.9)
    train_stems = new_stems[:split_idx]
    val_stems = new_stems[split_idx:]
    
    print(f"📊 Adding {len(train_stems)} to Train, {len(val_stems)} to Val.")

    # Helper to append lines
    def append_to_file(filename, stems):
        path = Path(dataset_dir) / filename
        with open(path, "a") as f:
            for stem in stems:
                # Assuming images are in 'images/' relative to dataset root or absolute
                # YOLO usually expects relative path from data.yaml location or absolute
                # Let's use absolute path for safety
                img_path = (Path(dataset_dir) / "images" / (stem + ".jpg")).absolute()
                f.write(f"{img_path}\n")

    if train_stems:
        append_to_file("train.txt", train_stems)
    if val_stems:
        append_to_file("val.txt", val_stems)
        
    print("✅ Split files updated.")

def main():
    dataset_dir = settings.DATASET_DIR
    
    # 1. Sync
    new_stems = sync_refined_data(dataset_dir)
    
    # 2. Update Splits
    update_split_files(dataset_dir, new_stems)

if __name__ == "__main__":
    main()
