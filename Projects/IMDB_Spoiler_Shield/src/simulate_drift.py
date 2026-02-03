import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter
from evidently.renderers.html import HtmlRenderer

def load_data():
    # Load processed data
    data_dir = "data/processed"
    bucket = os.getenv("S3_BUCKET")
    
    # Try local first
    if os.path.exists(os.path.join(data_dir, "train.csv")):
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    else:
        # Fallback to creating dummy data if running without full pipeline
        print("Warning: Processed data not found. Creating dummy data for simulation.")
        train_df = pd.DataFrame({
            'clean_review': ['text'] * 100,
            'label': np.random.choice([0, 1], 100),
            'avg_review_length': np.random.normal(500, 100, 100),
            'avg_sentiment_score': np.random.normal(0.5, 0.1, 100)
        })
        test_df = train_df.copy()

    # Ensure numerical columns exist (simulating if they came from Feast)
    if 'avg_review_length' not in train_df.columns:
        train_df['avg_review_length'] = np.random.normal(500, 100, len(train_df))
        test_df['avg_review_length'] = np.random.normal(500, 100, len(test_df))
    
    return train_df, test_df

def create_project(workspace):
    project = workspace.create_project("IMDB Spoiler Shield Monitoring")
    project.description = "Monitoring dashboard for Data Drift and Concept Drift simulations."
    project.save()
    return project

def simulate_drift():
    # 1. Setup Workspace
    workspace_path = "evidently_workspace"
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
    
    workspace = Workspace.create(workspace_path)
    
    # Check if project exists, else create
    projects = workspace.search_project("IMDB Spoiler Shield Monitoring")
    if len(projects) > 0:
        project = projects[0]
    else:
        project = create_project(workspace)

    print(f"Using Project: {project.name} (ID: {project.id})")

    # 2. Load Data
    ref_data, prod_data_base = load_data()
    
    # Define Column Mapping
    # Evidently needs to know what is what
    # We will focus on numerical drifts for visualization clarity
    
    # 3. Simulate Scenarios
    scenarios = [
        ("Batch 1: Normal Operation", prod_data_base.sample(frac=0.5, random_state=1)),
        
        ("Batch 2: Data Quality Drift (Null Lengths)", 
         prod_data_base.sample(frac=0.5, random_state=2).assign(
             avg_review_length=lambda x: x['avg_review_length'] * 0.01  # Simulate broken parser
         )),
         
        ("Batch 3: Concept Drift (Spoiler Attack)", 
         prod_data_base.sample(frac=0.5, random_state=3).assign(
             label=lambda x: 1  # Force all to be spoilers (1)
         ))
    ]

    # 4. Generate Reports
    for i, (scenario_name, current_data) in enumerate(scenarios):
        print(f"Running simulation: {scenario_name}...")
        
        # timestamp for the report
        report_ts = datetime.now() + timedelta(days=i)
        
        # Create Report
        report = Report(
            metrics=[
                DataDriftPreset(), 
                TargetDriftPreset(),
                DataQualityPreset()
            ],
            timestamp=report_ts
        )
        
        # Run calculation
        # We focus on specific columns to keep it clean
        cols = ['label', 'avg_review_length', 'avg_sentiment_score']
        # If columns don't exist in df (e.g. sentiment), add dummies
        for c in cols:
            if c not in ref_data.columns: ref_data[c] = 0.5
            if c not in current_data.columns: current_data[c] = 0.5

        report.run(
            reference_data=ref_data[cols],
            current_data=current_data[cols]
        )
        
        # Add to Workspace
        workspace.add_report(project.id, report)
        print(f"Report added for {scenario_name}")

    print("\nSimulation Complete!")
    print(f"To view dashboard: Run 'docker-compose up' and visit http://localhost:8001")

if __name__ == "__main__":
    simulate_drift()
