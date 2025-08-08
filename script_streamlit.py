import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from io import BytesIO

# Constants
boroughs = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']

# Agency-Complaint mapping
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

# Page configuration
st.set_page_config(
    page_title="Huduma Satisfaction Predictor",
    layout="wide"
)

# Title
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>Huduma Satisfaction Prediction System</h1>
""", unsafe_allow_html=True)

# Sidebar for navigation and description
st.sidebar.title("Navigation")
st.sidebar.markdown("""
Welcome to the Huduma Satisfaction Predictor.

Choose a mode below:
- **Single Prediction** for one complaint
- **Batch Prediction** to upload a CSV

Ensure the API is running at `http://localhost:8000`.
""")

option = st.sidebar.selectbox(
    "Choose a mode:",
    ["Single Prediction", "Batch Prediction (CSV Upload)"]
)

# API endpoint
api_url = "http://localhost:8000/predict"

def predict_single(complaint_text, agency_name, complaint_type, borough):
    try:
        payload = {
            "complaint_text": complaint_text,
            "agency_name": agency_name,
            "complaint_type": complaint_type,
            "borough": borough
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def predict_batch(df):
    results = []
    progress_bar = st.progress(0)

    for idx, row in df.iterrows():
        complaint_text = str(row.get('complaint_text', ''))
        agency_name = str(row.get('agency_name', 'Unknown'))
        complaint_type = str(row.get('complaint_type', 'Unknown'))
        borough = str(row.get('borough', 'Unknown'))

        result = predict_single(complaint_text, agency_name, complaint_type, borough)

        if 'error' not in result:
            results.append({
                'complaint_text': complaint_text,
                'agency_name': agency_name,
                'complaint_type': complaint_type,
                'borough': borough,
                'prediction': result.get('satisfaction_prediction', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'explanation': result.get('explanation', '')
            })
        else:
            results.append({
                'complaint_text': complaint_text,
                'agency_name': agency_name,
                'complaint_type': complaint_type,
                'borough': borough,
                'prediction': 'Error',
                'confidence': 0,
                'explanation': result['error']
            })

        progress_bar.progress((idx + 1) / len(df))

    return pd.DataFrame(results)

# Single Prediction 
if option == "Single Prediction":
    st.header("🔍 Single Complaint Prediction")

    st.markdown("""
    Fill out the complaint details below and click **Predict Satisfaction** to get the prediction.
    """)

    agency_name = st.sidebar.selectbox(
        "Select Agency:",
        list(agency_complaint_mapping.keys())
    )
    available_complaint_types = agency_complaint_mapping.get(agency_name, [])

    with st.form("prediction_form"):
        complaint_text = st.text_area(
            "📝 Complaint Text*",
            placeholder="Describe the complaint in detail...",
            height=150
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Agency Name**")
            st.write(agency_name)

        with col2:
            complaint_type = st.selectbox(
                "Complaint Type",
                available_complaint_types
            )

        with col3:
            borough = st.selectbox(
                "Borough/County",
                boroughs
            )

        submitted = st.form_submit_button("🔮 Predict Satisfaction")

    if submitted and complaint_text:
        with st.spinner("Making prediction..."):
            result = predict_single(complaint_text, agency_name, complaint_type, borough)

        if 'error' not in result:
            st.success("Prediction complete.")
            col1, col2, col3 = st.columns(3)

            with col1:
                prediction = result['satisfaction_prediction']
                color = "green" if prediction == "Satisfied" else "red"
                st.markdown(f"### Prediction: :{color}[{prediction}]")

            with col2:
                confidence = result['confidence']
                st.metric("Confidence", f"{confidence:.1%}")

            with col3:
                st.write("### Explanation")
                st.info(result['explanation'])
        else:
            st.error(result['error'])

    elif submitted:
        st.warning("Please enter complaint text to make a prediction.")

# Batch Prediction 
else:
    st.header("📁 Batch Prediction from CSV")

    st.info("""
    Upload a CSV file with complaints to get satisfaction predictions in bulk.

    **CSV Format Requirements:**
    - Required column: `complaint_text`
    - Optional columns: `agency_name`, `complaint_type`, `borough`
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv", help="Upload a CSV file with complaint data"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {len(df)} complaints.")

            with st.expander("📋 Preview Data", expanded=True):
                st.dataframe(df.head(5))

            if 'complaint_text' not in df.columns:
                st.error("❌ Missing required column: 'complaint_text'")
            else:
                if st.button("🚀 Generate Predictions", type="primary"):
                    with st.spinner("Processing predictions..."):
                        predictions_df = predict_batch(df)

                    st.success("✅ Predictions completed!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        satisfied_count = len(predictions_df[predictions_df['prediction'] == 'Satisfied'])
                        st.metric("Satisfied", satisfied_count)

                    with col2:
                        dissatisfied_count = len(predictions_df[predictions_df['prediction'] == 'Dissatisfied'])
                        st.metric("Dissatisfied", dissatisfied_count)

                    with col3:
                        avg_confidence = predictions_df['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

                    with st.expander("📊 Preview Predictions", expanded=True):
                        st.dataframe(predictions_df)

                    csv_buffer = BytesIO()
                    predictions_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_buffer.getvalue(),
                        file_name="satisfaction_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")