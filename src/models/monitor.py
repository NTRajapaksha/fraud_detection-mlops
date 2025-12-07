import pandas as pd
import mlflow
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_monitoring():
    # 1. Load Data
    try:
        reference = pd.read_csv("data/reference_data.csv")
        current = pd.read_csv("data/current_data.csv")
    except FileNotFoundError:
        print("❌ Error: Data files not found. Run train.py and simulate_traffic.py first.")
        return

    # 2. Configure MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Model_Drift_Monitoring")
    
    with mlflow.start_run(run_name="Weekly_Drift_Check"):
        print("📊 Calculating Data Drift...")
        
        # 3. Create Evidently Report
        # DataDriftPreset automatically checks all columns for statistical changes
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        
        # 4. Extract Metrics
        # We look for the "Dataset Drift" boolean score
        results = report.as_dict()
        drift_share = results['metrics'][0]['result']['share_of_drifted_columns']
        dataset_drift = results['metrics'][0]['result']['dataset_drift']
        
        # 5. Log to MLflow
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_param("dataset_drift_detected", dataset_drift)
        
        # 6. Save and Log Visual Report
        report_path = "drift_report.html"
        report.save_html(report_path)
        mlflow.log_artifact(report_path)
        
        print(f"✅ Monitoring Complete. Drift Detected: {dataset_drift}")
        print("📄 Report saved to MLflow artifacts.")
        
        # Cleanup local file
        if os.path.exists(report_path):
            os.remove(report_path)

if __name__ == "__main__":
    run_monitoring()