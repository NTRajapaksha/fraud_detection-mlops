import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

logging.basicConfig(level=logging.INFO)

def train_model(data_dir: str):
    mlflow.set_experiment("fraud-detection-v2")
    
    # Load Data
    X_train = pd.read_parquet(f"{data_dir}/X_train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/y_train.parquet").values.ravel()
    X_test = pd.read_parquet(f"{data_dir}/X_test.parquet")
    y_test = pd.read_parquet(f"{data_dir}/y_test.parquet").values.ravel()

    with mlflow.start_run():
        # Define Pipeline: Scale -> SMOTE -> Model
        # SMOTE only activates during fit(), not predict()
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss',
                random_state=42
            ))
        ])

        logging.info("Training pipeline...")
        pipeline.fit(X_train, y_train)

        # Evaluation
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        # Log params, metrics, and model
        mlflow.log_params(pipeline.named_steps['classifier'].get_params())
        mlflow.log_metrics(metrics)
        
        # Log the full pipeline (includes scaler!)
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="fraud_detection_pipeline",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        logging.info(f"Training Complete. Metrics: {metrics}")
        return pipeline

if __name__ == "__main__":
    train_model("data/processed")