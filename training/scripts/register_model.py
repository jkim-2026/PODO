import argparse
import mlflow
from mlflow import MlflowClient
import os
import sys
import boto3
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the model artifact (e.g. .onnx)")
    parser.add_argument("--tags", default="", help="Comma separated tags (key=value)")
    parser.add_argument("--model-name", default="PCB_Defect_Detector", help="Registered model name")
    args = parser.parse_args()

    # 1. Init MLflow (Using absolute path for reliability)
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    mlflow.set_experiment("PCB_Retraining_Pipeline")
    client = MlflowClient()

    print(f"📦 Registering model from: {args.model_path}")

    # Start a run for registration
    with mlflow.start_run(run_name="Model_Registration") as run:
        run_id = run.info.run_id

        # Log the ONNX as an artifact
        mlflow.log_artifact(args.model_path, artifact_path="model")

        # Log tags
        if args.tags:
            for tag in args.tags.split(","):
                if "=" in tag:
                    k, v = tag.split("=", 1)
                    mlflow.set_tag(k.strip(), v.strip())

    # 2. Register to Model Registry using the artifact URI from the completed run
    # Use the absolute artifact URI (not runs:/ shorthand which changed in MLflow 3.x)
    artifact_uri = client.get_run(run_id).info.artifact_uri
    model_source = f"{artifact_uri}/model/{os.path.basename(args.model_path)}"

    # Ensure registered model exists
    try:
        client.create_registered_model(args.model_name)
        print(f"📝 Created new registered model: '{args.model_name}'")
    except mlflow.exceptions.MlflowException:
        pass  # Already exists

    print(f"📝 Registering version to Model Registry as '{args.model_name}'...")
    version = client.create_model_version(
        name=args.model_name,
        source=model_source,
        run_id=run_id
    )

    print(f"✅ Model registered! Version: {version.version}")
    print(f"   Source Run ID: {run_id}")
    print(f"   Source URI:    {model_source}")

    # 3. Upload to S3 for Edge Devices
    s3_bucket = os.environ.get("S3_BUCKET_NAME", "pcb-data-storage")
    if not s3_bucket:
        print("⚠️ S3_BUCKET_NAME not set, skipping S3 upload.")
        return

    try:
        s3_client = boto3.client('s3')
        
        # Upload the ONNX file
        onnx_filename = os.path.basename(args.model_path)
        s3_key = f"models/candidates/{version.version}_{onnx_filename}"
        print(f"☁️ Uploading {args.model_path} to s3://{s3_bucket}/{s3_key}...")
        s3_client.upload_file(args.model_path, s3_bucket, s3_key)
        
        # Update latest.json
        import time
        latest_info = {
            "version": version.version,
            "s3_key": s3_key,
            "run_id": run_id,
            "timestamp": int(time.time() * 1000)
        }
        latest_json_path = "/tmp/latest.json"
        with open(latest_json_path, 'w') as f:
            json.dump(latest_info, f, indent=4)
            
        latest_s3_key = "models/candidates/latest.json"
        print(f"☁️ Updating s3://{s3_bucket}/{latest_s3_key}...")
        s3_client.upload_file(latest_json_path, s3_bucket, latest_s3_key)
        
        print("✅ S3 upload complete!")

    except Exception as e:
        print(f"❌ Failed to upload to S3: {e}")

if __name__ == "__main__":
    main()
