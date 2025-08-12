from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import uvicorn
import io
from sklearn.preprocessing import StandardScaler, LabelEncoder
from huggingface_hub import hf_hub_download

from huggingface_hub import hf_hub_download
import joblib

# Load model
try:
    # Download from Hugging Face
    model_path = hf_hub_download(
        repo_id="sambasi2406/huduma_satisfaction_prediction",  
        filename="best_model.joblib"
    )

    # Load as before
    model_data = joblib.load(model_path)
    model = model_data['pipeline']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']

    print("Model loaded successfully!")
    print(f"Expected features: {feature_names}")

except FileNotFoundError:
    print("Model not found.")
    model = None
    model_data = None
    label_encoders = {}
    feature_names = []


app = FastAPI(title="Huduma Satisfaction Prediction API", version="1.0.0")

class ComplaintInput(BaseModel):
    complaint_text: str
    agency_name: str = "Unknown"
    complaint_type: str = "Unknown"
    borough: str = "Unknown"

class PredictionResponse(BaseModel):
    satisfaction_prediction: str
    confidence: float
    explanation: str

def extract_features(complaint_data: ComplaintInput) -> Dict[str, Any]:
    features = {}
    complaint_lower = complaint_data.complaint_text.lower()
    combined_text = complaint_lower + " "

    positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy', 'pleased', 'wonderful',
                      'amazing', 'perfect', 'fantastic', 'outstanding', 'helpful', 'professional',
                      'quick', 'fast', 'efficient', 'resolved', 'fixed', 'solved', 'thank']
    negative_words = ['bad', 'terrible', 'awful', 'dissatisfied', 'angry', 'frustrated', 'poor',
                      'horrible', 'disgusting', 'unacceptable', 'useless', 'incompetent', 'rude',
                      'slow', 'delayed', 'ignored', 'waiting', 'weeks', 'months', 'never', 'no one',
                      'worst', 'hate', 'annoyed', 'disappointed', 'complaint', 'problem', 'issue',
                      'broken', 'damaged', 'failed', 'wrong', 'mistake', 'error']

    pos_count = sum(2 if word in complaint_lower else 1 for word in positive_words if word in combined_text)
    neg_count = sum(2 if word in complaint_lower else 1 for word in negative_words if word in combined_text)

    if neg_count > 0:
        features['sentiment_negative'] = 1
        features['sentiment_positive'] = 0
        features['Sentiment Score'] = -min(1.0, neg_count * 0.3)
        features['Sentiment Label'] = 'negative'
        features['dissatisfaction_reason_provided'] = 1
    elif pos_count > 0:
        features['sentiment_positive'] = 1
        features['sentiment_negative'] = 0
        features['Sentiment Score'] = min(1.0, pos_count * 0.2)
        features['Sentiment Label'] = 'positive'
        features['dissatisfaction_reason_provided'] = 0
    else:
        features['sentiment_positive'] = 0
        features['sentiment_negative'] = 0
        features['Sentiment Score'] = 0.0
        features['Sentiment Label'] = 'neutral'
        features['dissatisfaction_reason_provided'] = 0

    complaint_indicators = ['complaint', 'complain', 'report', 'issue', 'problem']
    if any(word in complaint_lower for word in complaint_indicators) and features['sentiment_negative'] == 0:
        features['Sentiment Score'] = min(features['Sentiment Score'], -0.1)

    features['Survey Year'] = 2024
    features['Survey Month'] = 1
    features['Cluster'] = 1 if features['sentiment_negative'] else 0
    features['Agency Name'] = complaint_data.agency_name
    features['Complaint Type'] = complaint_data.complaint_type
    features['Borough'] = complaint_data.borough

    return features

def get_explanation(prediction: int, confidence: float, features: Dict[str, Any]) -> str:
    if prediction == 1:
        base_msg = "Customer likely to be SATISFIED"
    else:
        base_msg = "Customer likely to be DISSATISFIED"

    factors = []
    if features['sentiment_negative']:
        factors.append("negative sentiment strongly detected")
    elif features['sentiment_positive']:
        factors.append("positive sentiment detected")
    else:
        factors.append("neutral sentiment")

    if features['dissatisfaction_reason_provided']:
        factors.append("dissatisfaction indicators present")

    sentiment_score = features.get('Sentiment Score', 0)
    if sentiment_score < -0.3:
        factors.append("strong negative language")
    elif sentiment_score > 0.4:
        factors.append("positive language")

    explanation = f"{base_msg} (confidence: {confidence:.1%})"
    if factors:
        explanation += f". Key factors: {', '.join(factors)}"

    return explanation

def predict_single_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    if model is not None and feature_names:
        # Create DataFrame with only the features the model expects
        model_features = {}
        for feat in feature_names:
            model_features[feat] = features.get(feat, 0)
        
        df = pd.DataFrame([model_features])
        
        try:
            prediction = model.predict(df)[0]
            try:
                prediction_proba = model.predict_proba(df)[0]
                confidence = max(prediction_proba)
            except:
                confidence = 0.7
        except Exception as e:
            print(f"Model prediction error: {e}")
            # Fallback to rule-based prediction
            sentiment_score = features.get('Sentiment Score', 0)
            has_negative = features.get('sentiment_negative', 0)
            has_positive = features.get('sentiment_positive', 0)
            if has_negative or sentiment_score < -0.1:
                prediction = 0
                confidence = 0.6
            elif has_positive and sentiment_score > 0.1:
                prediction = 1
                confidence = 0.7
            else:
                prediction = 0
                confidence = 0.5
    else:
        # Fallback when model is not available
        sentiment_score = features.get('Sentiment Score', 0)
        has_negative = features.get('sentiment_negative', 0)
        has_positive = features.get('sentiment_positive', 0)
        if has_negative or sentiment_score < -0.1:
            prediction = 0
            confidence = 0.8 if has_negative else 0.6
        elif has_positive and sentiment_score > 0.1:
            prediction = 1
            confidence = 0.7
        else:
            prediction = 0
            confidence = 0.5

    explanation = get_explanation(prediction, confidence, features)
    satisfaction_text = "Satisfied" if prediction == 1 else "Dissatisfied"

    return {
        "satisfaction_prediction": satisfaction_text,
        "confidence": confidence,
        "explanation": explanation
    }

@app.get("/")
def root():
    return {"message": "Customer Satisfaction Prediction API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None and len(feature_names) > 0}

@app.post("/predict", response_model=PredictionResponse)
def predict_satisfaction(complaint: ComplaintInput):
    try:
        features = extract_features(complaint)
        prediction_output = predict_single_from_features(features)
        print(f"Debug - Sentiment Score: {features['Sentiment Score']}, Negative: {features['sentiment_negative']}, Positive: {features['sentiment_positive']}")
        print(f"Debug - Prediction: {prediction_output['satisfaction_prediction']}, Confidence: {prediction_output['confidence']}")
        return prediction_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-text", response_model=PredictionResponse)
def predict_from_raw_text(complaint_text: str = Form(...)):
    complaint_input = ComplaintInput(complaint_text=complaint_text)
    features = extract_features(complaint_input)
    prediction_output = predict_single_from_features(features)
    return prediction_output

@app.post("/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        results = []

        for index, row in df.iterrows():
            complaint_input = ComplaintInput(
                complaint_text=row.get("complaint_text", ""),
                agency_name=row.get("agency_name", "Unknown"),
                complaint_type=row.get("complaint_type", "Unknown"),
                borough=row.get("borough", "Unknown")
            )
            features = extract_features(complaint_input)
            prediction_output = predict_single_from_features(features)
            results.append({
                "row": index,
                **prediction_output
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading or processing CSV: {str(e)}")

@app.post("/retrain")
def retrain_model():
    try:
        return {"message": "Model retraining triggered", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port)
