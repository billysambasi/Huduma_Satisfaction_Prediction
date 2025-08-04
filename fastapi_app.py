from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime

app = FastAPI(title="GoK Huduma Satisfaction Predictor", version="1.0.0")

# Global variables for model reloading
model = None
encodings = None
model_loaded_time = None

def load_model():
    global model, encodings, model_loaded_time
    try:
        model = joblib.load('satisfaction_model.pkl')
        encodings = joblib.load('encodings.pkl')
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
    descriptor: str
    borough: str
    year: int
    month: int

@app.get("/")
def read_root():
    return {
        "message": "NYC 311 Satisfaction Predictor API",
        "model_loaded": model is not None,
        "loaded_at": model_loaded_time
    }

@app.post("/reload-model")
def reload_model():
    """Reload model without restarting server - useful for model iteration"""
    success = load_model()
    return {
        "success": success,
        "loaded_at": model_loaded_time,
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    }

@app.get("/options")
def get_options():
    try:
        if not encodings:
            return {"error": "Encodings not loaded"}
        
        # Convert numpy values to regular Python strings
        agencies = [str(key) for key in encodings['Agency Name'].keys()]
        complaint_types = [str(key) for key in encodings['Complaint Type'].keys()]
        descriptors = [str(key) for key in encodings['Descriptor'].keys()]
        boroughs = [str(key) for key in encodings['Borough'].keys()]
        
        return {
            "agencies": agencies,
            "complaint_types": complaint_types,
            "descriptors": descriptors,
            "boroughs": boroughs
        }
    except Exception as e:
        return {"error": f"Options error: {str(e)}"}


@app.post("/predict")
def predict_satisfaction(request: PredictionRequest):
    if not model or not encodings:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_data = pd.DataFrame({
            'Agency Name': [encodings['Agency Name'][request.agency_name]],
            'Complaint Type': [encodings['Complaint Type'][request.complaint_type]],
            'Descriptor': [encodings['Descriptor'][request.descriptor]],
            'Borough': [encodings['Borough'][request.borough]],
            'Survey Year': [request.year],
            'Survey Month': [request.month]
        })
        
        prediction = model.predict_proba(input_data)[0][1]
        return {
            "satisfaction_probability": round(prediction, 4),
            "model_loaded_at": model_loaded_time
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encodings_loaded": encodings is not None
    }
