import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import logging
from textblob import TextBlob
import altair as alt

# Page Config
st.set_page_config(page_title="Huduma Satisfaction Predictor", layout="wide")

# Logging Setup
logging.basicConfig(filename="app.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load Model
@st.cache_resource
def load_model():
    try:
        if os.path.exists('best_model.joblib'):
            return joblib.load('best_model.joblib')
        elif os.path.exists('satisfaction_model.pkl'):
            with open('satisfaction_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            st.error("❌ No model file found. Please ensure 'best_model.joblib' or 'satisfaction_model.pkl' exists.")
            return None
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
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
            'probability': probability[1],
            'confidence': max(probability)
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {'error': str(e)}

# Main App
def main():
    st.title("🏛️ Huduma Satisfaction Predictor")
    st.markdown("Predict citizen satisfaction with NYC 311 service requests")

    model_data = load_model()
    if model_data is None:
        st.stop()

    st.success("✅ Model loaded successfully!")

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

                required_cols = {'agency_name', 'complaint_type', 'borough', 'year', 'month', 'complaint_text'}
                if not required_cols.issubset(df.columns):
                    st.error("CSV missing required columns.")
                elif st.button("🚀 Generate Predictions"):
                    results_df = df.apply(
                        lambda row: pd.Series(make_prediction(
                            model_data,
                            row['agency_name'],
                            row['complaint_type'],
                            row['borough'],
                            row['year'],
                            row['month'],
                            row['complaint_text']
                        )),
                        axis=1
                    )
                    final_df = pd.concat([df, results_df], axis=1)
                    st.success("✅ Predictions completed!")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Satisfied", (final_df['prediction'] == 'Satisfied').sum())
                    col2.metric("Dissatisfied", (final_df['prediction'] == 'Dissatisfied').sum())
                    col3.metric("Avg Confidence", f"{final_df['confidence'].mean():.1%}")

                    # Visualization
                    chart = alt.Chart(final_df).mark_bar().encode(
                        x='prediction',
                        y='count()',
                        color='prediction'
                    )
                    st.altair_chart(chart, use_container_width=True)

                    st.dataframe(final_df)

                    # Download button
                    csv = final_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "satisfaction_predictions.csv", "text/csv")

            except Exception as e:
                logging.error(f"Batch processing error: {str(e)}")
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()