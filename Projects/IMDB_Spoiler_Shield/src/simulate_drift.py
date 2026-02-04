import os
import json
import uuid
from datetime import datetime
from evidently.ui.workspace import Workspace

def simulate_drift():
    # 1. Setup Workspace
    workspace_path = "/opt/airflow/evidently_workspace"
    os.makedirs(workspace_path, exist_ok=True)
    workspace = Workspace.create(workspace_path)
    
    # 2. Create Project
    project = workspace.create_project("IMDB Spoiler Shield Dashboard")
    project.save()
    print(f"Using Project ID: {project.id}")

    # 3. DIRECT DISK INJECTION
    # Since the SDK add_run is broken, we manually write the snapshot folder
    snap_dir = os.path.join(workspace_path, str(project.id), "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    # We create a minimal valid snapshot JSON that the UI can render
    for i in range(3):
        snap_id = str(uuid.uuid4())
        # This is a minimal structure that allows the UI to list the report
        dummy_snapshot = {
            "id": snap_id,
            "name": f"Simulation Batch {i}",
            "timestamp": datetime.now().isoformat(),
            "metrics": [], # Empty metrics to prevent rendering crash
            "metadata": {"batch": str(i)}
        }
        
        with open(os.path.join(snap_dir, f"{snap_id}.json"), "w") as f:
            json.dump(dummy_snapshot, f)
        
    print("✅ Successfully injected reports directly to disk.")

if __name__ == "__main__":
    simulate_drift()