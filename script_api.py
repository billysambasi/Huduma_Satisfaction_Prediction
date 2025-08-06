from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import uvicorn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model
try:
    model = joblib.load('best_model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found. Using rule-based prediction.")
    model = None

# Define features and mappings
categorical_features = ['Agency Name', 'Complaint Type', 'Borough', 'Sentiment Label']
numerical_features = ['Survey Year', 'Survey Month', 'Cluster', 'Sentiment Score', 
                     'resolution_length', 'has_dissatisfaction_reason', 
                     'sentiment_positive', 'sentiment_negative']

# Initialize basic encoders and scaler (will be fitted on-the-fly)
scaler = StandardScaler()
label_encoders = {}
for feat in categorical_features:
    label_encoders[feat] = LabelEncoder()

app = FastAPI(title="Customer Satisfaction Prediction API", version="1.0.0")

class ComplaintInput(BaseModel):
    complaint_text: str
    agency_name: str = "Unknown"
    complaint_type: str = "Unknown"
    borough: str = "Unknown"
    resolution_description: str = ""

class PredictionResponse(BaseModel):
    satisfaction_prediction: str
    confidence: float
    explanation: str

def extract_features(complaint_data: ComplaintInput) -> Dict[str, Any]:
    """Extract features from complaint input"""
    
    features = {}
    
    # Enhanced sentiment analysis
    complaint_lower = complaint_data.complaint_text.lower()
    resolution_lower = complaint_data.resolution_description.lower() if complaint_data.resolution_description else ""
    combined_text = complaint_lower + " " + resolution_lower
    
    # Expanded keyword lists
    positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy', 'pleased', 'wonderful', 
                     'amazing', 'perfect', 'fantastic', 'outstanding', 'helpful', 'professional',
                     'quick', 'fast', 'efficient', 'resolved', 'fixed', 'solved', 'thank']
    
    negative_words = ['bad', 'terrible', 'awful', 'dissatisfied', 'angry', 'frustrated', 'poor',
                     'horrible', 'disgusting', 'unacceptable', 'useless', 'incompetent', 'rude',
                     'slow', 'delayed', 'ignored', 'waiting', 'weeks', 'months', 'never', 'no one',
                     'worst', 'hate', 'annoyed', 'disappointed', 'complaint', 'problem', 'issue',
                     'broken', 'damaged', 'failed', 'wrong', 'mistake', 'error']
    
    # Count sentiment words with weights
    pos_count = sum(2 if word in complaint_lower else 1 for word in positive_words if word in combined_text)
    neg_count = sum(2 if word in complaint_lower else 1 for word in negative_words if word in combined_text)
    
    # Determine sentiment with stronger negative bias
    if neg_count > 0:
        features['sentiment_negative'] = 1
        features['sentiment_positive'] = 0
        features['Sentiment Score'] = -min(1.0, neg_count * 0.3)  # Stronger negative scoring
        features['Sentiment Label'] = 'negative'
        features['has_dissatisfaction_reason'] = 1  # Assume dissatisfaction if negative sentiment
    elif pos_count > 0:
        features['sentiment_positive'] = 1
        features['sentiment_negative'] = 0
        features['Sentiment Score'] = min(1.0, pos_count * 0.2)
        features['Sentiment Label'] = 'positive'
        features['has_dissatisfaction_reason'] = 0
    else:
        features['sentiment_positive'] = 0
        features['sentiment_negative'] = 0
        features['Sentiment Score'] = 0.0
        features['Sentiment Label'] = 'neutral'
        features['has_dissatisfaction_reason'] = 0
    
    # Additional negative indicators
    complaint_indicators = ['complaint', 'complain', 'report', 'issue', 'problem']
    if any(word in complaint_lower for word in complaint_indicators) and features['sentiment_negative'] == 0:
        features['Sentiment Score'] = min(features['Sentiment Score'], -0.1)  # Slight negative bias for complaints
    
    # Other features
    features['Survey Year'] = 2024
    features['Survey Month'] = 1
    features['Cluster'] = 1 if features['sentiment_negative'] else 0
    features['resolution_length'] = len(complaint_data.resolution_description) if complaint_data.resolution_description else 50
    
    # Categorical features
    features['Agency Name'] = complaint_data.agency_name
    features['Complaint Type'] = complaint_data.complaint_type
    features['Borough'] = complaint_data.borough
    
    return features

def get_explanation(prediction: int, confidence: float, features: Dict[str, Any]) -> str:
    """Generate explanation for the prediction"""
    
    if prediction == 1:
        base_msg = "Customer likely to be SATISFIED"
    else:
        base_msg = "Customer likely to be DISSATISFIED"
    
    # Key factors
    factors = []
    
    if features['sentiment_negative']:
        factors.append("negative sentiment strongly detected")
    elif features['sentiment_positive']:
        factors.append("positive sentiment detected")
    else:
        factors.append("neutral sentiment")
    
    if features['has_dissatisfaction_reason']:
        factors.append("dissatisfaction indicators present")
    
    sentiment_score = features.get('Sentiment Score', 0)
    if sentiment_score < -0.3:
        factors.append("strong negative language")
    elif sentiment_score > 0.3:
        factors.append("positive language")
    
    if features['resolution_length'] > 200:
        factors.append("detailed resolution provided")
    elif features['resolution_length'] < 50:
        factors.append("minimal resolution details")
    
    explanation = f"{base_msg} (confidence: {confidence:.1%})"
    if factors:
        explanation += f". Key factors: {', '.join(factors)}"
    
    return explanation

@app.get("/")
async def root():
    return {"message": "Customer Satisfaction Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_satisfaction(complaint: ComplaintInput):
    """Predict customer satisfaction from complaint text"""
    
    try:
        # Extract features
        features = extract_features(complaint)
        
        if model is not None:
            # ML prediction
            feature_vector = []
            
            # Add numerical features
            for feat in numerical_features:
                feature_vector.append(features[feat])
            
            # Add encoded categorical features (use hash for unseen categories)
            for feat in categorical_features:
                value = features[feat]
                encoded_value = hash(value) % 100  # Simple hash encoding
                feature_vector.append(encoded_value)
            
            # Convert to numpy array
            X = np.array(feature_vector).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            try:
                prediction_proba = model.predict_proba(X)[0]
                confidence = max(prediction_proba)
            except:
                confidence = 0.7
        else:
            # Fallback rule-based prediction
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
        
        # Generate explanation
        explanation = get_explanation(prediction, confidence, features)
        
        # Map prediction to text
        satisfaction_text = "Satisfied" if prediction == 1 else "Dissatisfied"
        
        # Debug info (remove in production)
        print(f"Debug - Sentiment Score: {features['Sentiment Score']}, Negative: {features['sentiment_negative']}, Positive: {features['sentiment_positive']}")
        print(f"Debug - Prediction: {prediction}, Confidence: {confidence}")
        
        return PredictionResponse(
            satisfaction_prediction=satisfaction_text,
            confidence=confidence,
            explanation=explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/retrain")
async def retrain_model():
    """Endpoint to trigger model retraining"""
    try:
        # In a real scenario, this would retrain the model with new data
        # For now, just return a success message
        return {"message": "Model retraining triggered", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)