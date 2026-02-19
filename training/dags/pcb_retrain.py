from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

# Project settings
PROJECT_ROOT = "/workspace/final_project/training"
SCRIPTS_DIR = f"{PROJECT_ROOT}/scripts"
MODEL_CONFIG = f"{PROJECT_ROOT}/configs/config_qat.yaml"

default_args = {
    'owner': 'podo_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pcb_retrain_pipeline',
    default_args=default_args,
    description='PCB Defect Detection Retraining Pipeline (Sync -> Train/QAT -> Export)',
    schedule_interval='0 18 * * 0',  # Every Sunday at 18:00 UTC (Mon 03:00 KST)
    start_date=days_ago(1),
    tags=['podo', 'qat', 'mlops'],
    catchup=False
) as dag:

    # 1. Sync Data from S3
    t1_sync = BashOperator(
        task_id='sync_data_from_s3',
        bash_command=f'python {SCRIPTS_DIR}/sync_data.py',
        cwd=PROJECT_ROOT
    )

    # 2. Train QAT Model (Includes Recalibration)
    # Using 'train_qat.py' which we modified to be robust
    t2_train = BashOperator(
        task_id='train_qat_model',
        bash_command=f'python {SCRIPTS_DIR}/train_qat.py --config {MODEL_CONFIG}',
        cwd=PROJECT_ROOT
    )

    # 3. Export to ONNX
    # We assume training saves to runs/qat/default/weights/best.pt (or similar)
    # We need to dynamically find the latest 'best.pt' or 'best_qat_fallback.pt'
    # For simplicity in this DAG, we'll try to target the expected path.
    # A better approach would be passing the run ID, but here we assume 'latest' logic in export script or consistent naming.
    # Let's assume train_qat.py logs the final path. OR we use a fixed path structure for automation.
    
    # We will use a wrapper command to find the latest run and export it.
    export_cmd = f"""
    LATEST_RUN=$(ls -td {PROJECT_ROOT}/runs/qat/*/ | head -1)
    BEST_PT="${{LATEST_RUN}}weights/best.pt"
    # Check if fallback exists
    if [ -f "${{LATEST_RUN}}weights/best_qat_fallback.pt" ]; then
        BEST_PT="${{LATEST_RUN}}weights/best_qat_fallback.pt"
    fi
    
    echo "Exporting ${{BEST_PT}}..."
    python {SCRIPTS_DIR}/export_qat.py --weights "${{BEST_PT}}" --base-weights {PROJECT_ROOT}/yolo11n.pt --full-int8 --output "${{LATEST_RUN}}weights/best.onnx"
    """

    t3_export = BashOperator(
        task_id='export_onnx',
        bash_command=export_cmd,
        cwd=PROJECT_ROOT
    )
    
    # 4. Register to MLflow (and update S3 latest.json)
    # We can create a small script for this or inline it.
    # Let's assume we have a script 'register_model.py'
    register_cmd = f"""
    LATEST_RUN=$(ls -td {PROJECT_ROOT}/runs/qat/*/ | head -1)
    ONNX_PATH="${{LATEST_RUN}}weights/best.onnx"
    
    python {SCRIPTS_DIR}/register_model.py --model-path "${{ONNX_PATH}}" --tags "stage=staging"
    """
    
    t4_register = BashOperator(
        task_id='register_model',
        bash_command=register_cmd,
        cwd=PROJECT_ROOT
    )

    t1_sync >> t2_train >> t3_export >> t4_register
