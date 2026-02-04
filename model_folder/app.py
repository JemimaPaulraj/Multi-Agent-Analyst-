# app.py
import os
import tarfile
from pathlib import Path
from datetime import date, datetime, timedelta

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

app = FastAPI()

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "ticket-forecasting-lake")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "Model/model.tar.gz")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Local paths
MODEL_DIR = Path("/opt/ml/model")
MODEL_PATH = MODEL_DIR / "model.joblib"

_model = None


class InvokePayload(BaseModel):
    horizon_days: int = Field(..., ge=1, le=365)
    start_date: str | None = None  # "YYYY-MM-DD"


def download_model_from_s3():
    """Download and extract model from S3."""
    print(f"[MODEL] Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = MODEL_DIR / "model.tar.gz"
    
    # Download from S3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(tar_path))
    print(f"[MODEL] Downloaded to {tar_path}")
    
    # Extract tar.gz
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=MODEL_DIR)
    print(f"[MODEL] Extracted to {MODEL_DIR}")
    
    # Clean up tar file
    tar_path.unlink()
    
    # List extracted files
    for f in MODEL_DIR.iterdir():
        print(f"[MODEL] Found: {f}")


def load_model():
    global _model
    if _model is None:
        # Download from S3 if not exists locally
        if not MODEL_PATH.exists():
            download_model_from_s3()
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH} after download")
        
        _model = joblib.load(MODEL_PATH)
        print(f"[MODEL] Loaded from {MODEL_PATH}")
    return _model


@app.get("/ping")
def ping():
    # SageMaker health check endpoint
    try:
        load_model()
        return {"status": "ok"}
    except Exception as e:
        # return non-200 => endpoint considered unhealthy
        return Response(content=str(e), status_code=500)


@app.post("/invocations")
def invocations(payload: InvokePayload):
    # Parse start_date
    if payload.start_date:
        try:
            start = datetime.strptime(payload.start_date, "%Y-%m-%d").date()
        except ValueError:
            start = date.today()
    else:
        start = date.today()

    # Build future dataframe
    future_dates = [start + timedelta(days=i) for i in range(payload.horizon_days)]
    future_df = pd.DataFrame({"ds": pd.to_datetime(future_dates)})

    model = load_model()
    pred = model.predict(future_df)

    forecast = []
    for i, row in pred.iterrows():
        forecast.append(
            {
                "day": int(i) + 1,
                "date": row["ds"].strftime("%Y-%m-%d"),
                "forecast_ticket_count": int(round(row["yhat"])),
                "lower_bound": int(round(row["yhat_lower"])),
                "upper_bound": int(round(row["yhat_upper"])),
            }
        )

    return {"forecast": forecast}
