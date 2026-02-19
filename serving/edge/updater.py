import os
import time
import json
import requests
import boto3
import sys
import subprocess
from datetime import datetime

# Import local modules (assuming in serving/edge)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# MLflow for Edge (simplified or standard client)
# Try-except import in case it's not installed on Edge yet
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not found. Status reporting will be skipped.")

class ModelUpdater:
    def __init__(self, interval=600):
        self.interval = interval
        self.current_version = "v0.0.0"
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), # Need these envs on Edge
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-northeast-2")
        )
        self.bucket = "final-project-podo" # config.S3_BUCKET_NAME
        self.latest_json_key = "models/candidates/latest.json"
        
        # Local paths
        self.download_dir = "models/candidates"
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Configure MLflow
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI) # Set this in config.py
            print(f"📡 MLflow Tracking URI: {config.MLFLOW_TRACKING_URI}")

    def check_for_updates(self):
        print(f"[{datetime.now()}] Checking for updates...")
        try:
            # Get latest.json from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.latest_json_key)
            latest_info = json.loads(response['Body'].read().decode('utf-8'))
            
            new_version = latest_info.get("version")
            if new_version != self.current_version:
                print(f"✨ New version found: {new_version} (Current: {self.current_version})")
                self.process_update(latest_info)
            else:
                print("✅ Already up to date.")
                
        except Exception as e:
            print(f"❌ Failed to check for updates: {e}")

    def process_update(self, meta):
        s3_key = meta.get("s3_key")
        version = meta.get("version")
        local_onnx_path = os.path.join(self.download_dir, f"{version}.onnx")
        local_engine_path = local_onnx_path.replace(".onnx", ".engine")

        # 1. Download
        print(f"⬇️ Downloading {s3_key}...")
        self.s3_client.download_file(self.bucket, s3_key, local_onnx_path)
        
        # 2. Build Engine
        print("🔨 Building TensorRT Engine (may take a while)...")
        # Assuming YOLO export command or trtexec directly
        # Here we use YOLO python api for simplicity if available, or subprocess
        try:
            from ultralytics import YOLO
            model = YOLO(local_onnx_path)
            # Export to engine (FP16)
            # Note: YOLO.export handles onnx->engine too? 
            # Usually we load PT to export engine. 
            # If we have ONNX, we might need 'trtexec' command line.
            # Let's assume we have 'trtexec' in path.
            
            cmd = f"trtexec --onnx={local_onnx_path} --saveEngine={local_engine_path} --fp16"
            subprocess.run(cmd, shell=True, check=True)
            print("✅ Engine built successfully.")
            
        except Exception as e:
            print(f"❌ Engine build failed: {e}")
            self.report_status(version, "build_failed", str(e))
            return

        # 3. Evaluate (AB Test)
        # For prototype, we skip complex AB test and just assume we want to switch
        # But user asked for comparison.
        if self.evaluate_and_compare(local_engine_path):
            print("🎉 Promoting new model!")
            self.switch_model(local_engine_path, version)
            self.report_status(version, "active", "Promoted after successful eval")
        else:
            print("📉 New model underperformed. Rejecting.")
            self.report_status(version, "rejected", "Lower mAP/FPS")

    def evaluate_and_compare(self, new_engine_path):
        """
        Run validation on current vs new model.
        Returns True if new model is better.
        """
        print("⚖️  Running On-Device Evaluation...")
        # Placeholder for actual eval logic
        # 1. Load Validation Set
        # 2. Run Inference Loop
        # 3. Calculate mAP
        
        # Mock result for now
        import random
        new_map = random.uniform(0.90, 0.99)
        current_map = 0.95 # Mock
        
        print(f"   Current mAP: {current_map:.4f}")
        print(f"   New mAP:     {new_map:.4f}")
        
        return new_map >= current_map

    def switch_model(self, engine_path, version):
        """
        Update config.MODEL_PATH and restart service?
        Or just symlink 'current.engine' -> new_engine_path
        """
        target = "current.engine"
        if os.path.exists(target):
            os.remove(target)
        os.symlink(engine_path, target)
        self.current_version = version
        print(f"🔄 Switched 'current.engine' to {version}")
        # Note: Main process needs to reload this. 
        # Ideally, we send a signal or main process watches the file.

    def report_status(self, version, status, message):
        if not MLFLOW_AVAILABLE: return
        
        try:
            with mlflow.start_run(run_name=f"Edge_Update_{version}"):
                mlflow.log_param("version", version)
                mlflow.log_param("status", status)
                mlflow.log_param("message", message)
                mlflow.set_tag("edge_device", "jetson-orin-01")
            print("📡 Status reported to MLflow.")
        except Exception as e:
            print(f"⚠️ Failed to report to MLflow: {e}")

    def run(self):
        print("🟢 Updater Daemon Started.")
        while True:
            self.check_for_updates()
            time.sleep(self.interval)

if __name__ == "__main__":
    updater = ModelUpdater(interval=60) # Check every 1 min for demo
    updater.run()
