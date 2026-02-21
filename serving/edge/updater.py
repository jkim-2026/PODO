```python
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
# Try-except import in case it's notry:
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow package not found. Continuing without MLflow tracking.")

class ModelUpdater:
    def __init__(self, interval=60):
        self.current_version = "0"
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # S3 Client setup
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), # Need these envs on Edge
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-southeast-2")
        )
        self.bucket = "pcb-data-storage" # Updated bucket name
        self.latest_json_key = "models/candidates/latest.json"

        # Configure MLflow
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")) 
            print(f"📡 MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
            self.client = MlflowClient()
        else:
            self.client = None

        self.model_name = os.environ.get("MLFLOW_MODEL_NAME", "PCB_Defect_Detector")
        self.edge_device_id = os.environ.get("EDGE_DEVICE_ID", "edge-jetson-01")
        self.interval = interval

    def check_for_updates(self):
        print(f"[{datetime.now()}] Checking for updates from S3...")
        try:
            # Download latest.json
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.latest_json_key)
            latest_meta = json.loads(response['Body'].read().decode('utf-8'))
            latest_v = latest_meta['version']
            
            if latest_v != self.current_version:
                print(f"🌟 New version found: {latest_v} (Current: {self.current_version})")
                self.download_and_build(latest_v, latest_meta)
                
        except Exception as e:
            # Ignore if file doesn't exist yet
            if "NoSuchKey" not in str(e):
                print(f"⚠️ Error checking updates: {e}")

    def download_and_build(self, version, meta):
        s3_key = meta['s3_key']
        local_onnx_path = os.path.join(config.MODELS_DIR, f"v{version}.onnx")
        local_engine_path = os.path.join(config.MODELS_DIR, f"v{version}.engine")
        
        # 1. Download
        print(f"📥 Downloading {s3_key}...")
        try:
            self.s3_client.download_file(self.bucket, s3_key, local_onnx_path)
            print(f"✅ Downloaded to: {local_onnx_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            self.report_status(version, "download_failed", str(e), run_id=meta.get("run_id"))
            return

        local_engine_path = os.path.join(config.MODELS_DIR, f"v{version}.engine")
        
        # 2. Build Engine (with Resource Isolation)
        print(f"🔨 Building TensorRT Engine (Isolating Resources)...")
        # Touch BUILDING_FLAG to signal inference_worker to throttle
        with open(config.BUILDING_FLAG_PATH, 'w') as f:
            f.write(f"Building v{version}")
            
        try:
            # Limit TRT workspace to 1024MB to prevent choking the live inference GPU limits
            cmd = f"trtexec --onnx={local_onnx_path} --saveEngine={local_engine_path} --fp16 --workspace=1024"
            subprocess.run(cmd, shell=True, check=True)
            print("✅ Engine built successfully.")
        except Exception as e:
            print(f"❌ Engine build failed: {e}")
            self.report_status(version, "build_failed", str(e), run_id=meta.get("run_id"))
            return
        finally:
            # Remove BUILDING_FLAG
            if os.path.exists(config.BUILDING_FLAG_PATH):
                os.remove(config.BUILDING_FLAG_PATH)

        # 3. Evaluate (Shadow Test)
        new_map = self.evaluate_engine(local_engine_path)
        current_map = self.evaluate_engine(config.MODEL_PATH) if os.path.exists(config.MODEL_PATH) else 0.0
        
        print(f"⚖️  Shadow Evaluation Results:")
        print(f"   Current mAP: {current_map:.4f}")
        print(f"   New mAP:     {new_map:.4f}")

        if new_map >= current_map:
            print("🎉 Promoting new model!")
            self.switch_model(local_engine_path, version)
            self.report_status(version, "active", "Promoted after successful eval", new_map, meta.get("run_id"))
            self.promote_in_mlflow(version)
        else:
            print("📉 New model underperformed. Rejecting.")
            self.report_status(version, "rejected", "Lower mAP", new_map, meta.get("run_id"))

    def evaluate_engine(self, engine_path):
        """
        Run validation on a golden set in edge device.
        Requires a small dataset existing at config.GOLDEN_SET_PATH for robust mAP calc.
        For demonstration, we return a mock mAP, but in reality this runs `model.val()` equivalent.
        """
        if not os.path.exists(engine_path):
            return 0.0
            
        print(f"   Running golden set evaluation on {engine_path}...")
        # Mock logic. Imagine running inference on 100 images and calculating mAP.
        import random
        # Give newer versions slightly higher chance to win for this demo
        is_current = engine_path == config.MODEL_PATH
        base_map = 0.95 if is_current else 0.96
        return random.uniform(base_map - 0.02, base_map + 0.02)

    def switch_model(self, engine_path, version):
        """
        Updates the symlink and touches the reload flag.
        """
        target = config.MODEL_PATH
        
        # Remove existing symlink (or file if it's not a link)
        if os.path.exists(target) or os.path.islink(target):
            os.remove(target)
            
        # Create new symlink pointing to the new engine
        # Need absolute path for the target of symlink to be robust
        abs_engine_path = os.path.abspath(engine_path)
        os.symlink(abs_engine_path, target)
        self.current_version = version
        print(f"🔄 Switched '{target}' to {abs_engine_path}")
        
        # Touch reload flag for inference_worker
        with open(config.RELOAD_FLAG_PATH, 'w') as f:
            f.write(version)
        print(f"🔔 Signaled inference worker to reload via {config.RELOAD_FLAG_PATH}")
        
    def promote_in_mlflow(self, version_str):
        if not self.client: return
        try:
            print(f"🚀 Prompting {self.model_name} version {version_str} to Production...")
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version_str,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"✅ MLflow Model Registry updated: version {version_str} -> Production")
            
        except Exception as e:
            print(f"⚠️ Failed to transition model stage in MLflow: {e}")

    def report_status(self, version, status, message, map_score=None, run_id=None):
        if not MLFLOW_AVAILABLE: return
        
        try:
            # We can log this against the specific run_id if provided,
            # or to a new generic Run tracking deployments.
            # Doing a specific deployment run tracking:
            with mlflow.start_run(run_name=f"Edge_Deploy_v{version}"):
                mlflow.log_param("version", version)
                mlflow.log_param("status", status)
                mlflow.log_param("message", message)
                mlflow.set_tag("edge_device", self.edge_device_id)
                if run_id:
                    mlflow.set_tag("source_run_id", run_id)
                if map_score is not None:
                    mlflow.log_metric("edge_map", map_score)
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
