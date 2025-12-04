from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
import pandas as pd
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Global model variable
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    model_uri = os.getenv("MODEL_URI", "models:/fraud-detection-v2@production")
    try:
        logger.info(f"Loading model from {model_uri}")
        ml_models["model"] = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        ml_models["model"] = None
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

class TransactionInput(BaseModel):
    # Dynamic generation of fields based on PCA V1-V28 + Time + Amount
    features: list[float] = Field(
        ..., 
        min_length=30, 
        max_length=30, 
        description="[Time, V1...V28, Amount]"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 30 
            }
        }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": ml_models["model"] is not None}

@app.post("/predict")
async def predict(transaction: TransactionInput):
    if not ml_models["model"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame with generic columns to match training schema expectation
        # (Assuming columns are Time, V1..V28, Amount)
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        input_df = pd.DataFrame([transaction.features], columns=cols)
        
        # Pipeline handles scaling automatically
        prediction = ml_models["model"].predict(input_df)
        probs = ml_models["model"].predict_proba(input_df)
        
        is_fraud = bool(prediction[0])
        confidence = float(probs[0][1]) if is_fraud else float(probs[0][0])
        
        return {
            "is_fraud": is_fraud,
            "fraud_probability": float(probs[0][1]),
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")