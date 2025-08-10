import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px


# Configurations and Constants

st.set_page_config(page_title="Huduma Satisfaction Dashboard", layout="wide")

API_URL = "http://127.0.0.1:8000"

boroughs = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']

agency_complaint_mapping = {
    'Department of Buildings': [
        'Adult Establishment', 'Advertising Sign', 'Building Condition',
        'Elevator Issue', 'Illegal Conversion', 'Construction Noise', 'Facade Crack'
    ],
    'NYPD': [
        'Illegal Parking', 'Noise', 'Traffic Signal Condition',
        'Loitering', 'Suspicious Activity', 'Unlicensed Vendor', 'Disturbance'
    ],
    'Department of Sanitation': [
        'Dirty Conditions', 'Missed Collection', 'Recycling Enforcement',
        'Overflowing Trash', 'Blocked Trash Bin', 'Illegal Dumping', 'Littering'
    ],
    'Department of Transportation': [
        'Street Condition', 'Traffic Signal Condition', 'Street Light Condition',
        'Pothole', 'Faded Crosswalk', 'Broken Sign', 'Obstructed View'
    ],
    'Department for the Aging': [
        'Legal Services Provider Complaint', 'Senior Center Complaint',
        'Inadequate Meals', 'Transportation Delay', 'Abuse Report', 'Service Denial', 'Late Pickup'
    ]
}

if "history" not in st.session_state:
    st.session_state.history = []

# Helper Functions 

def predict_single(payload):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def predict_text_only(text):
    try:
        response = requests.post(f"{API_URL}/predict-text", data={"complaint_text": text})
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def predict_batch(df):
    results = []
    progress_bar = st.progress(0)
    for idx, row in df.iterrows():
        payload = {
            "complaint_text": str(row.get('complaint_text', '')),
            "agency_name": str(row.get('agency_name', 'Unknown')),
            "complaint_type": str(row.get('complaint_type', 'Unknown')),
            "borough": str(row.get('borough', 'Unknown'))
        }
        result = predict_single(payload)
        if 'error' not in result:
            results.append({
                **payload,
                'prediction': result.get('satisfaction_prediction', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'explanation': result.get('explanation', '')
            })
        else:
            results.append({
                **payload,
                'prediction': 'Error',
                'confidence': 0,
                'explanation': result['error']
            })
        progress_bar.progress((idx + 1) / max(1, len(df)))
    return pd.DataFrame(results)


# User Interface - Tabs

st.title("Huduma Satisfaction Prediction Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Complaint", "💬 Text Only", "📂 CSV Upload", "📊 Dashboard"
])

# Tab 1: Single Complaint 

with tab1:

    complaint_text = st.text_area("📝 Complaint Text*", height=150)

    col1, col2, col3 = st.columns(3)

    with col1:
        agency_name = st.selectbox("Agency Name", list(agency_complaint_mapping.keys()))

    with col2:
        complaint_type = st.selectbox("Complaint Type", agency_complaint_mapping[agency_name])

    with col3:
        borough = st.selectbox("Borough/County", boroughs)

    if st.button("🔍 Predict (Single Complaint)", key="predict_single_btn"):
        payload = {
            "complaint_text": complaint_text,
            "agency_name": agency_name,
            "complaint_type": complaint_type,
            "borough": borough
        }
        result = predict_single(payload)
        if 'error' not in result:
            st.success(f"Prediction: {result['satisfaction_prediction']}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.info(f"Explanation: {result['explanation']}")
            st.session_state.history.append({
                "Mode": "Single Complaint",
                "Prediction": result['satisfaction_prediction'],
                "Confidence": result['confidence']
            })
        else:
            st.error(result['error'])

# Tab 2: Text Only

with tab2:
    text_only = st.text_area("Enter Complaint Text Only", height=200)
    if st.button("🔍 Predict (Text Only)", key="predict_text_btn"):
        result = predict_text_only(text_only)
        if 'error' not in result:
            st.success(f"Prediction: {result['satisfaction_prediction']}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.info(f"Explanation: {result['explanation']}")
            st.session_state.history.append({
                "Mode": "Text Only",
                "Prediction": result['satisfaction_prediction'],
                "Confidence": result['confidence']
            })
        else:
            st.error(result['error'])

# Tab 3: CSV Upload 

with tab3:
    st.info("Upload CSV with at least `complaint_text`. Optional: `agency_name`, `complaint_type`, `borough`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {len(df)} complaints.")
            st.dataframe(df.head(5))
            if 'complaint_text' not in df.columns:
                st.error("❌ Missing required column: 'complaint_text'")
            else:
                if st.button("🚀 Generate Predictions", key="predict_csv_btn"):
                    predictions_df = predict_batch(df)
                    st.success("✅ Predictions completed!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Satisfied", len(predictions_df[predictions_df['prediction'] == 'Satisfied']))
                    with col2:
                        st.metric("Dissatisfied", len(predictions_df[predictions_df['prediction'] == 'Dissatisfied']))
                    with col3:
                        st.metric("Avg Confidence", f"{predictions_df['confidence'].mean():.1%}")
                    st.dataframe(predictions_df)
                    csv_buffer = BytesIO()
                    predictions_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_buffer.getvalue(),
                        file_name="satisfaction_predictions.csv",
                        mime="text/csv"
                    )
                    for _, row in predictions_df.iterrows():
                        st.session_state.history.append({
                            "Mode": "CSV Upload",
                            "Prediction": row['prediction'],
                            "Confidence": row['confidence']
                        })
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# Tab 4: Dashboard 

with tab4:
    st.header("📈 Prediction Summary")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        pie_fig = px.pie(
            history_df,
            names="Prediction",
            title="Prediction Share",
            hole=0.3
        )
    
        st.plotly_chart(pie_fig, use_container_width=True)

    else:
        st.info("No predictions yet.")

    st.markdown("---")
    if st.button("🔄 Retrain Model", key="retrain_btn"):
        try:
            response = requests.post(f"{API_URL}/retrain")
            if response.status_code == 200:
                st.success(response.json().get("message", "Model retrained successfully."))
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
