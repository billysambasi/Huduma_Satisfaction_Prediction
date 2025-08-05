from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from satisfaction_pipeline import SatisfactionPipeline
from datetime import datetime

app = FastAPI(title="GoK Huduma Satisfaction Predictor API", version="1.0.0")

# Global variables
model = None
model_loaded_time = None

def load_model():
    global model, model_loaded_time
    try:
        model = SatisfactionPipeline.load_model('satisfaction_model.pkl')
        model_loaded_time = datetime.now()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
load_model()

class PredictionRequest(BaseModel):
    agency_name: str
    complaint_type: str
    borough: str
    year: int
    month: int
    cluster: int
    sentiment_score: float

@app.get("/")
def read_root():
    return {
        "message": "NYC 311 Satisfaction Predictor API",
        "model_loaded": model is not None,
        "loaded_at": model_loaded_time
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/reload-model")
def reload_model():
    success = load_model()
    return {
        "success": success,
        "loaded_at": model_loaded_time,
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    }

@app.post("/predict")
def predict_satisfaction(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_data = pd.DataFrame({
            'Agency Name': [request.agency_name],
            'Complaint Type': [request.complaint_type],
            'Borough': [request.borough],
            'Survey Year': [request.year],
            'Survey Month': [request.month],
            'Cluster': [request.cluster],
            'Sentiment Score': [request.sentiment_score]
        })
        
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        return {
            "satisfaction_probability": round(prediction_proba, 4),
            "prediction": int(prediction),
            "prediction_label": "Satisfied" if prediction == 1 else "Not Satisfied",
            "model_loaded_at": model_loaded_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")