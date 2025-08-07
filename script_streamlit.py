import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from io import BytesIO

# Constants
BOROUGHS = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']

# Agency-Complaint mapping
AGENCY_COMPLAINT_MAPPING = {
    'Department of Buildings': ['Adult Establishment', 'Advertising Sign', 'Building Condition'],
    'NYPD': ['Illegal Parking', 'Noise', 'Traffic Signal Condition'],
    'Department of Sanitation': ['Dirty Conditions', 'Missed Collection', 'Recycling Enforcement'],
    'Department of Transportation': ['Street Condition', 'Traffic Signal Condition', 'Street Light Condition'],
    'Department for the Aging': ['Legal Services Provider Complaint', 'Senior Center Complaint']
}

# Page config
st.set_page_config(
    page_title="Huduma Satisfaction Predictor",
    layout="wide"
)

# Title
st.title("Huduma Satisfaction Prediction System")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Single Prediction", "Batch Prediction (CSV Upload)"]
)

# API endpoint
API_URL = "http://localhost:8000/predict"

def predict_single(complaint_text, agency_name, complaint_type, borough):
    """Make single prediction via API"""
    try:
        payload = {
            "complaint_text": complaint_text,
            "agency_name": agency_name,
            "complaint_type": complaint_type,
            "borough": borough
        }
        
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def predict_batch(df):
    """Make batch predictions for DataFrame"""
    results = []
    progress_bar = st.progress(0)
    
    for idx, row in df.iterrows():
        # Extract data from row
        complaint_text = str(row.get('complaint_text', ''))
        agency_name = str(row.get('agency_name', 'Unknown'))
        complaint_type = str(row.get('complaint_type', 'Unknown'))
        borough = str(row.get('borough', 'Unknown'))
        
        # Make prediction
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
        
        # Update progress
        progress_bar.progress((idx + 1) / len(df))
    
    return pd.DataFrame(results)

if option == "Single Prediction":
    st.header("🔍 Single Complaint Prediction")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            complaint_text = st.text_area(
                "Complaint Text*",
                placeholder="Enter the complaint description...",
                height=150
            )
            
            agency_name = st.selectbox(
                "Agency Name",
                list(AGENCY_COMPLAINT_MAPPING.keys())
            )
        
        with col2:
            # Dynamic complaint type based on selected agency
            available_complaint_types = AGENCY_COMPLAINT_MAPPING.get(agency_name, [])
            complaint_type = st.selectbox(
                "Complaint Type",
                available_complaint_types
            )
            
            borough = st.selectbox(
                "Borough",
                BOROUGHS
            )
        
        submitted = st.form_submit_button("🔮 Predict Satisfaction")
    
    if submitted and complaint_text:
        with st.spinner("Making prediction..."):
            result = predict_single(complaint_text, agency_name, complaint_type, borough)
        
        if 'error' not in result:
            # Display results
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
                st.write(result['explanation'])
        else:
            st.error(result['error'])
    
    elif submitted:
        st.warning("Please enter complaint text to make a prediction.")

else:  # Batch Prediction
    st.header("📁 Batch Prediction from CSV")
    
    # Instructions
    st.info("""
    **CSV Format Requirements:**
    - Required column: `complaint_text`
    - Optional columns: `agency_name`, `complaint_type`, `borough`
    - Each row represents one complaint to predict
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with complaint data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Display file info
            st.success(f"✅ File uploaded successfully! Found {len(df)} complaints to process.")
            
            # Show preview
            with st.expander("📋 Preview Data", expanded=True):
                st.dataframe(df.head(10))
            
            # Validate required columns
            if 'complaint_text' not in df.columns:
                st.error("❌ Missing required column: 'complaint_text'")
            else:
                # Process predictions
                if st.button("🚀 Generate Predictions", type="primary"):
                    with st.spinner("Processing predictions... This may take a while for large files."):
                        predictions_df = predict_batch(df)
                    
                    st.success("✅ Predictions completed!")
                    
                    # Display results summary
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
                    
                    # Preview results
                    with st.expander("📊 Preview Predictions", expanded=True):
                        st.dataframe(predictions_df)
                    
                    # Download button
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

# Footer
st.markdown("---")
st.markdown("**Note:** Make sure the API server is running on `http://localhost:8000` before making predictions.")