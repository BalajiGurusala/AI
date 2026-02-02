from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add src to sys.path
sys.path.append(os.path.join(os.environ.get("AIRFLOW_HOME", "/opt/airflow"), "src"))

from data_loader import load_data
from preprocessing import preprocess_data
from train import train_model
from evaluate import evaluate_model
from feature_engineering import generate_features
from compare_models import compare_models
from create_dataset_with_feast import create_dataset
from train_advanced_models import main as train_advanced_main
from train_bert import run_training as train_bert_main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'imdb_spoiler_pipeline',
    default_args=default_args,
    description='End-to-end MLOps pipeline for IMDB Spoiler Shield with Feast and Ray',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # 1. Ingest Data
    data_ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=load_data,
        op_kwargs={'data_dir': '/opt/airflow/data'}
    )

    # 2. Process Text Data (Baseline & BERT)
    data_processing_task = PythonOperator(
        task_id='data_processing',
        python_callable=preprocess_data,
        op_kwargs={'data_dir': '/opt/airflow/data', 'output_dir': '/opt/airflow/data/processed'}
    )

    # 3. Feature Engineering (Feast Materialization)
    feature_engineering_task = PythonOperator(
        task_id='feature_engineering_store',
        python_callable=generate_features,
        op_kwargs={'data_dir': '/opt/airflow/data', 'output_dir': '/opt/airflow/data/processed'}
    )

    # 4. Create Feast Training Dataset (Joins Processed Data + Historical Features)
    create_feast_dataset_task = PythonOperator(
        task_id='create_feast_dataset',
        python_callable=create_dataset
    )

    # 5a. Train Baseline Model (Logistic Regression)
    model_training_task = PythonOperator(
        task_id='train_baseline_model',
        python_callable=train_model,
        op_kwargs={'data_dir': '/opt/airflow/data/processed'}
    )

    # 5b. Train Advanced Models (XGBoost/NN with Feast Features)
    train_advanced_task = PythonOperator(
        task_id='train_advanced_models',
        python_callable=train_advanced_main
    )

    # 5c. Train BERT Model (Distributed/Ray)
    train_bert_task = PythonOperator(
        task_id='train_bert_model',
        python_callable=train_bert_main
    )

    # 6. Evaluate Baseline Model
    model_evaluation_task = PythonOperator(
        task_id='evaluate_baseline_model',
        python_callable=evaluate_model,
        op_kwargs={'data_dir': '/opt/airflow/data/processed', 'model_path': '/opt/airflow/models/model.joblib'}
    )

    # 7. Compare All Models
    model_comparison_task = PythonOperator(
        task_id='compare_models',
        python_callable=compare_models
    )

    # --- Dependencies ---

    # Ingestion feeds Processing and Feature Engineering
    data_ingestion_task >> data_processing_task
    data_ingestion_task >> feature_engineering_task

    # Baseline Path
    data_processing_task >> model_training_task >> model_evaluation_task

    # Feast Path: Needs both Processed Data (for IDs/Labels) and Feature Store (for Stats)
    [data_processing_task, feature_engineering_task] >> create_feast_dataset_task
    create_feast_dataset_task >> train_advanced_task

    # BERT Path: Needs Processed Data
    data_processing_task >> train_bert_task

    # Convergence: Compare results after all training/eval is done
    [model_evaluation_task, train_advanced_task, train_bert_task] >> model_comparison_task