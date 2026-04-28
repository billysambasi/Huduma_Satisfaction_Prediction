import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from textblob import TextBlob
import altair as alt
from huggingface_hub import hf_hub_download

# Page Config
st.set_page_config(page_title="Huduma Satisfaction Predictor", layout="wide")

# Load Model from Hugging Face Hub
@st.cache_resource
def load_model():
    try:
        # Always fetch from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id="sambasi2406/huduma_satisfaction_predictor",
            filename="best_model.joblib"
        )
        model_data = joblib.load(model_path)

        # Validate model structure
        if not all(key in model_data for key in ['pipeline', 'label_encoders']):
            st.error("❌ Invalid model format. Expected keys: pipeline, label_encoders")
            return None

        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Constants
BOROUGHS = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']

AGENCY_COMPLAINTS = {
    'Department of Buildings': ['Building Condition', 'Elevator Issue', 'Illegal Conversion'],
    'NYPD': ['Illegal Parking', 'Noise', 'Traffic Signal Condition'],
    'Department of Sanitation': ['Dirty Conditions', 'Missed Collection', 'Recycling Enforcement'],
    'Department of Transportation': ['Street Condition', 'Pothole', 'Street Light Condition'],
    'Department for the Aging': ['Legal Services Provider Complaint', 'Senior Center Complaint']
}

# Helper Functions
def get_sentiment_score(text):
    if not text or pd.isna(text):
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

def get_cluster(text):
    if not text or pd.isna(text):
        return 2
    return len(str(text)) % 5

def make_prediction(model_data, agency, complaint_type, borough, year=2024, month=6, text=""):
    try:
        features = pd.DataFrame({
            'Agency Name': [agency],
            'Complaint Type': [complaint_type],
            'Borough': [borough],
            'Survey Year': [year],
            'Survey Month': [month],
            'Cluster': [get_cluster(text)],
            'Sentiment Score': [get_sentiment_score(text)]
        })

        pipeline = model_data['pipeline']
        label_encoders = model_data['label_encoders']

        for col in ['Agency Name', 'Complaint Type', 'Borough']:
            if col in label_encoders:
                try:
                    features[col] = label_encoders[col].transform(features[col].astype(str))
                except ValueError:
                    features[col] = 0

        prediction = pipeline.predict(features)[0]
        probability = pipeline.predict_proba(features)[0]

        return {
            'prediction': 'Satisfied' if prediction == 1 else 'Dissatisfied',
            'probability': float(probability[1]),
            'confidence': float(max(probability))
        }

    except Exception as e:
        return {'error': str(e)}

# Main App
def main():
    st.title("🏛️ Huduma Satisfaction Predictor")
    st.markdown("Predict citizen satisfaction with NYC 311 service requests")

    model_data = load_model()
    if model_data is None:
        st.stop()

    st.success("✅ Model loaded successfully from Hugging Face Hub!")

    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Upload"])

    # Single Prediction
    with tab1:
        st.header("Single Complaint Prediction")

        col1, col2 = st.columns(2)
        with col1:
            agency = st.selectbox("Agency", list(AGENCY_COMPLAINTS.keys()))
            complaint_type = st.selectbox("Complaint Type", AGENCY_COMPLAINTS[agency])
            borough = st.selectbox("Borough", BOROUGHS)
        with col2:
            year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
            month = st.number_input("Month", min_value=1, max_value=12, value=6)

        complaint_text = st.text_area("Complaint Description (optional)", height=100)

        if st.button("🔮 Predict Satisfaction", type="primary"):
            result = make_prediction(model_data, agency, complaint_type, borough, year, month, complaint_text)

            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", result['prediction'])
                col2.metric("Satisfaction Probability", f"{result['probability']:.1%}")
                col3.metric("Confidence", f"{result['confidence']:.1%}")

                if result['prediction'] == 'Satisfied':
                    st.success("😊 Citizen likely to be satisfied")
                else:
                    st.warning("😞 Citizen likely to be dissatisfied")

    # Batch Upload
    with tab2:
        st.header("Batch Prediction from CSV")
        st.info("Upload CSV with columns: agency_name, complaint_type, borough, year, month, complaint_text")

        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:", df.head())

                if st.button("🚀 Generate Predictions"):
                    results = []
                    progress = st.progress(0)

                    for idx, row in df.iterrows():
                        result = make_prediction(
                            model_data,
                            row.get('agency_name', 'NYPD'),
                            row.get('complaint_type', 'Noise'),
                            row.get('borough', 'MANHATTAN'),
                            row.get('year', 2024),
                            row.get('month', 6),
                            row.get('complaint_text', '')
                        )
                        results.append(result)
                        progress.progress((idx + 1) / len(df))

                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

                    st.success("✅ Predictions completed!")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    satisfied_count = (final_df['prediction'] == 'Satisfied').sum()
                    dissatisfied_count = (final_df['prediction'] == 'Dissatisfied').sum()
                    avg_confidence = final_df['confidence'].mean()

                    col1.metric("Satisfied", satisfied_count)
                    col2.metric("Dissatisfied", dissatisfied_count)
                    col3.metric("Avg Confidence", f"{avg_confidence:.1%}")

                    st.dataframe(final_df)

                    # Download button
                    csv = final_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "satisfaction_predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()