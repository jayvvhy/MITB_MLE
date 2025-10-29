"""
Airflow DAG for monthly loan model lifecycle pipeline:
- Monthly ETL (Bronze → Silver → Gold)
- Conditional model retraining (pre/post deployment)
- One-time model deployment
- Monthly inference and monitoring
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
import os
import json
import random

# ======================
# 🧩 CONFIGURABLE PARAMETERS
# ======================

DEPLOYMENT_DATE = datetime(2024, 6, 1)
FIRST_TRAINING_DATE = datetime(2024, 4, 1)
MONITORING_START_DATE = datetime(2024, 10, 1)
MODEL_REGISTRY_PATH = "/opt/airflow/models"
ALERT_THRESHOLD = 0.70  # Example: AUC threshold for alert

# ======================
# 🧠 HELPER FUNCTIONS
# ======================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# --- Simulated functions for illustration ---
def run_etl_layer(layer, **context):
    """Simulate ETL process for bronze/silver/gold layers."""
    run_date = context['ds']
    print(f"Running {layer} ETL for snapshot {run_date} ...")

def train_candidate_model(**context):
    """Train and save a candidate model."""
    run_date = context['ds']
    model_dir = os.path.join(MODEL_REGISTRY_PATH, "candidate_models", run_date)
    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, f"model_{run_date}.pkl")

    # Simulate training and performance metric
    auc = round(random.uniform(0.6, 0.9), 3)
    with open(model_path, "w") as f:
        f.write(f"Dummy model - AUC: {auc}\n")

    print(f"✅ Candidate model saved: {model_path}, AUC={auc}")
    # Save metadata
    metadata = {"run_date": run_date, "auc": auc}
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

def deploy_best_model(**context):
    """Pick best model from candidate_models and deploy."""
    candidate_root = os.path.join(MODEL_REGISTRY_PATH, "candidate_models")
    deployed_dir = os.path.join(MODEL_REGISTRY_PATH, "deployed_model")
    ensure_dir(deployed_dir)

    # Find all candidate metadata
    best_model, best_auc = None, 0
    for month in sorted(os.listdir(candidate_root)):
        meta_path = os.path.join(candidate_root, month, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta["auc"] > best_auc:
                best_auc = meta["auc"]
                best_model = month

    if best_model:
        deployed_meta = {
            "deployed_model": f"model_{best_model}.pkl",
            "deployment_date": str(DEPLOYMENT_DATE.date()),
            "auc": best_auc,
        }
        with open(os.path.join(deployed_dir, "deployment_metadata.json"), "w") as f:
            json.dump(deployed_meta, f)
        print(f"🚀 Deployed best model from {best_model} with AUC={best_auc}")
    else:
        print("⚠️ No candidate models found for deployment!")

def run_inference(**context):
    """Run inference using deployed model."""
    run_date = context['ds']
    deployed_meta_path = os.path.join(MODEL_REGISTRY_PATH, "deployed_model", "deployment_metadata.json")
    if not os.path.exists(deployed_meta_path):
        print("⚠️ No deployed model found, skipping inference.")
        return

    with open(deployed_meta_path, "r") as f:
        deployed_meta = json.load(f)

    print(f"Running inference for {run_date} using {deployed_meta['deployed_model']} ...")
    # In real setup: load model and predict → save to gold_predictions folder
    pred_dir = os.path.join(MODEL_REGISTRY_PATH, "predictions", run_date)
    ensure_dir(pred_dir)
    with open(os.path.join(pred_dir, "predictions.csv"), "w") as f:
        f.write("Dummy predictions\n")

def monitor_models(**context):
    """Monitor deployed model performance."""
    run_date = context['ds']
    report_dir = os.path.join(MODEL_REGISTRY_PATH, "monitoring_reports", run_date)
    ensure_dir(report_dir)

    # Simulate metric computation (AUC)
    auc = round(random.uniform(0.5, 0.9), 3)
    report_path = os.path.join(report_dir, "monitoring_report.json")
    with open(report_path, "w") as f:
        json.dump({"date": run_date, "auc": auc}, f)
    print(f"📊 Monitoring report generated: {report_path}, AUC={auc}")

    if auc < ALERT_THRESHOLD:
        raise ValueError(f"❌ ALERT: Model AUC dropped to {auc} (< {ALERT_THRESHOLD})")

# ======================
# ⚙️ BRANCH LOGIC FUNCTIONS
# ======================

def determine_phase(execution_date, **kwargs):
    if execution_date < FIRST_TRAINING_DATE:
        return "skip_training"
    elif execution_date < DEPLOYMENT_DATE:
        return "model_training_pre"
    elif execution_date == DEPLOYMENT_DATE:
        return "deploy_best"
    elif execution_date > DEPLOYMENT_DATE:
        return "model_training_post"
    return "skip_training"

def should_monitor(execution_date, **kwargs):
    return "run_monitoring" if execution_date >= MONITORING_START_DATE else "skip_monitoring"

# ======================
# 🚀 DEFINE DAG
# ======================

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "loan_model_lifecycle",
    default_args=default_args,
    description="Monthly ML model lifecycle DAG (ETL, training, deployment, inference, monitoring)",
    schedule_interval="0 0 1 * *",  # Every 1st of month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,
    max_active_runs=1,
    tags=["ml", "model_lifecycle"],
) as dag:

bronze_layer = BashOperator(
    task_id='bronze_layer',
    bash_command=(
        'cd /opt/airflow/scripts && '
        'python3 data_processing_bronze_table.py "{{ ds }}" "config/bronze_config.yaml"'
    ),
)

silver_layer = BashOperator(
    task_id="run_silver_layer",
    bash_command=(
        "cd /opt/airflow/scripts && "
        "python3 data_processing_silver_table.py {{ ds }} config/silver_config.yaml"
    ),
)

with DAG(
    dag_id="monthly_model_pipeline",
    start_date=datetime(2024, 4, 1),
    schedule_interval="@monthly",
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,  # wrapper for model_train.py
    )

    select_and_deploy_task = PythonOperator(
        task_id="select_best_model",
        python_callable=select_best_model,
    )

    inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference,  # wrapper for model_inference.py
    )

    # Conditional: Only run selection/deployment at 2024-06-01
    check_deployment_date = ShortCircuitOperator(
        task_id="check_deployment_date",
        python_callable=is_deployment_date,
    )

    # DAG dependencies
    train_task >> check_deployment_date >> select_and_deploy_task >> inference_task

# Implement a model_registry.csv as central registry file

promote_task = PythonOperator(task_id="promote_best_model", python_callable=promote_main)
train_task = PythonOperator(task_id="train_model", python_callable=train_main)
inference_task = PythonOperator(task_id="run_inference", python_callable=infer_main)
monitor_task = PythonOperator(task_id="monitor_model", python_callable=monitor_main)

promote_task >> train_task >> inference_task >> monitor_task


    
    # --- ETL Layer ---
    etl_bronze = PythonOperator(task_id="etl_bronze", python_callable=run_etl_layer, op_args=["bronze"])
    etl_silver = PythonOperator(task_id="etl_silver", python_callable=run_etl_layer, op_args=["silver"])
    etl_gold = PythonOperator(task_id="etl_gold", python_callable=run_etl_layer, op_args=["gold"])
    etl_bronze >> etl_silver >> etl_gold

    # --- Branch for Training / Deployment ---
    branch_phase = BranchPythonOperator(task_id="branch_phase", python_callable=determine_phase)
    etl_gold >> branch_phase

    skip_training = DummyOperator(task_id="skip_training")
    model_training_pre = PythonOperator(task_id="model_training_pre", python_callable=train_candidate_model)
    deploy_best = PythonOperator(task_id="deploy_best", python_callable=deploy_best_model)
    model_training_post = PythonOperator(task_id="model_training_post", python_callable=train_candidate_model)

    branch_phase >> [skip_training, model_training_pre, deploy_best, model_training_post]

    # --- Inference ---
    inference = PythonOperator(task_id="inference", python_callable=run_inference)
    [deploy_best, model_training_post] >> inference

    # --- Monitoring ---
    branch_monitor = BranchPythonOperator(task_id="branch_monitor", python_callable=should_monitor)
    run_monitoring = PythonOperator(task_id="run_monitoring", python_callable=monitor_models)
    skip_monitoring = DummyOperator(task_id="skip_monitoring")

    inference >> branch_monitor
    branch_monitor >> [run_monitoring, skip_monitoring]

    # --- Alert on Failure ---
    alert_email = EmailOperator(
        task_id="alert_email",
        to="alerts@example.com",
        subject="Model Monitoring Alert",
        html_content="Model performance degraded. Please review candidate models.",
        trigger_rule="one_failed",  # triggers only if previous task failed
    )

    run_monitoring >> alert_email

