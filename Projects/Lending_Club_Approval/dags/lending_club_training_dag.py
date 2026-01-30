from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Project configuration
# Adjust PROJECT_HOME in your environment or use the fallback
project_dir = os.environ.get("PROJECT_HOME", "/Users/balaji.gurusala/Documents/Code/AI_Coding/AI/Projects/Lending_Club_Approval")

# Define the DAG
with DAG(
    'lending_club_training_pipeline',
    default_args=default_args,
    description='Orchestrates Lending Club Feature Sync and Model Tuning',
    schedule_interval='@weekly',
    catchup=False,
    tags=['lending_club', 'mlops', 'feast', 'ray'],
) as dag:

    # Task 1: feature_store_sync
    # Syncs the online store (SQLite/Redis) with the latest S3/Parquet data
    feature_store_sync = BashOperator(
        task_id='feature_store_sync',
        bash_command=f'cd {project_dir}/feature_repo && '
                     f'../.venv/bin/feast materialize-incremental $(date +%Y-%m-%d)',
    )

    # Task 2: model_tuning
    # Triggers the Ray Tune + MLflow optimization job
    model_tuning = BashOperator(
        task_id='model_tuning',
        bash_command=f'cd {project_dir} && '
                     'export PYTHONPATH=$PYTHONPATH:. && '
                     'export UPLOAD_TO_S3=true && ' # Trigger S3 persistence
                     '.venv/bin/python src/train_ray.py',
    )

    # Set dependency: Sync features before starting the tuning job
    feature_store_sync >> model_tuning
