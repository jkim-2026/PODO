import argparse
import mlflow
import json
import boto3
import os
import sys
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Basic config
# In a real scenario, import from config.py
S3_BUCKET_NAME = "final-project-podo"
S3_MODEL_PREFIX = "models/candidates"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the model artifact (e.g. .onnx or .pt)")
    parser.add_argument("--tags", default="", help="Comma separated tags (key=value)")
    args = parser.parse_args()

    # 1. Init MLflow
    mlflow.set_tracking_uri("file:./mlruns") # Or remote
    mlflow.set_experiment("PCB_Retraining_Pipeline")

    print(f"📦 Registering model: {args.model_path}")
    
    # Start a run for registration
    with mlflow.start_run(run_name="Model_Registration") as run:
        # Log artifact
        mlflow.log_artifact(args.model_path)
        
        # Log tags
        if args.tags:
            for tag in args.tags.split(","):
                k, v = tag.split("=")
                mlflow.set_tag(k, v)
        
        print(f"✅ Model registered to MLflow Run: {run.info.run_id}")

        # 2. Upload to S3 for Edge
        s3 = boto3.client("s3")
        filename = os.path.basename(args.model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{S3_MODEL_PREFIX}/{timestamp}_{filename}"
        
        try:
            s3.upload_file(args.model_path, S3_BUCKET_NAME, s3_key)
            print(f"☁️  Uploaded to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
            
            # 3. Update 'latest.json'
            latest_info = {
                "version": timestamp,
                "url": f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}", # Or Presigned
                "s3_key": s3_key,
                "created_at": datetime.now().isoformat(),
                "run_id": run.info.run_id
            }
            
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{S3_MODEL_PREFIX}/latest.json",
                Body=json.dumps(latest_info, indent=2),
                ContentType="application/json"
            )
            print("📝 Updated latest.json")
            
        except Exception as e:
            print(f"❌ Failed to upload/update S3: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
