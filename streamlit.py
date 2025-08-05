import streamlit as st
import pandas as pd
from satisfaction_pipeline import SatisfactionPipeline

# Load the trained model
@st.cache_resource
def load_model():
    return SatisfactionPipeline.load_model('satisfaction_model.pkl')

model = load_model()

st.title('GoK Huduma Satisfaction Predictor')
st.write('Predict citizen satisfaction with GoK Huduma requests')

# Input fields
col1, col2 = st.columns(2)

with col1:
    agency = st.selectbox('Agency Name', [
        'Department of Buildings', 'Department for the Aging', 'Department of Parks and Recreation',
        'Department of Environmental Protection', 'Department of Sanitation', 'Department of Housing Preservation and Development',
        'New York City Police Department', 'Department of Health and Mental Hygiene', 'Department of Consumer and Worker Protection',
        'Department of Finance', 'Department of Transportation', 'Department of Homeless Services', 'Economic Development Corporation'
    ])
    
    complaint_type = st.selectbox('Complaint Type', [
        'Adult Establishment', 'Legal Services Provider Complaint', 'Advertising Sign', 'Root/Sewer/Sidewalk Condition',
        'New Tree Request', 'Debris', 'Water Maintenance', 'Illegal Tree Damage', 'Illegal Dumping', 'Homeless Person Assistance',
        'General', 'Obstruction', 'Paint/Plaster', 'Lead', 'Noise', 'Elevator', 'DOB Posted Notice or Order',
        'Missed Collection', 'Abandoned Bike', 'Building Condition', 'No Building Permit', 'Construction In Progress',
        'Illegal Conversion', 'Light from Parking Lot', 'Blocked Driveway', 'DCWP Literature Request', 'Indoor Air Quality',
        'DOF Property - Request Copy', 'Building Exit', 'Boiler', 'Unsanitary Condition', 'Curb Cut/Driveway',
        'Overgrown Tree/Branches', 'Electrical Wiring', 'Certificate of Occupancy', 'Rodent', 'Sewer Maintenance',
        'Consumer Complaint', 'Illegal Parking', 'Abandoned Vehicle', 'Street Sign - Damaged', 'Heat/Hot Water',
        'Damaged Tree', 'Street Sign - Dangling', 'Encampment', 'Sidewalk Condition', 'Dead/Dying Tree',
        'Street Condition', 'Outdoor Dining', 'Smoking', 'Food Establishment', 'Bike/Roller/Skate Chronic',
        'Non-Emergency Police Matter', 'Snow or Ice', 'DOF Property - Reduction Issue', 'Litter Basket Request',
        'DOF Literature Request', 'Unsanitary Animal Pvt Property', 'DCA Literature Request', 'Awning/Canopy/Marquee',
        'Curb Condition', 'DOF Property - Update Account', 'DOF Property - RPIE Issue', 'Abandoned Building',
        'Residential Disposal Complaint', 'Illegal Use', 'Water Conservation', 'Plumbing Work in Past',
        'Building Sign', 'Unstable Building', 'Scaffold/Sidewalk Shed', 'Gas Hookup/ Piping', 'Water Drainage',
        'Building Sprinkler System', 'Air Quality', 'Appliance', 'Electric', 'Door/Window', 'Flooring/Stairs',
        'Safety', 'Water Leak', 'Plumbing', 'Home Delivered Meal - Missed Delivery', 'Case Management Agency Complaint',
        'Noise - Commercial', 'Noise - Residential', 'Noise - Street/Sidewalk', 'Noise - Park', 'Noise - Vehicle',
        'Noise - Helicopter'
    ])
    
    borough = st.selectbox('Borough', ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND'])

with col2:
    year = st.number_input('Survey Year', min_value=2020, max_value=2025, value=2024)
    month = st.number_input('Survey Month', min_value=1, max_value=12, value=6)
    cluster = st.selectbox('Cluster', [0, 1, 2, 3, 4])
    sentiment_score = st.slider('Sentiment Score', min_value=-1.0, max_value=1.0, value=0.0, step=0.01)

if st.button('Predict Satisfaction', type='primary'):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Agency Name': [agency],
        'Complaint Type': [complaint_type],
        'Borough': [borough],
        'Survey Year': [year],
        'Survey Month': [month],
        'Cluster': [cluster],
        'Sentiment Score': [sentiment_score]
    })
    
    try:
        # Make prediction
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success('Prediction Complete!')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Satisfaction Probability', f'{prediction_proba:.1%}')
        with col2:
            satisfaction_label = 'Satisfied' if prediction == 1 else 'Not Satisfied'
            st.metric('Prediction', satisfaction_label)
        
        # Progress bar for probability
        st.progress(prediction_proba)
        
        # Interpretation
        if prediction_proba >= 0.7:
            st.success('High likelihood of citizen satisfaction')
        elif prediction_proba >= 0.5:
            st.warning('Moderate likelihood of citizen satisfaction')
        else:
            st.error('Low likelihood of citizen satisfaction')
            
    except Exception as e:
        st.error(f'Error making prediction: {str(e)}')

# Model info
with st.expander('Model Information'):
    st.write("""
    This model predicts citizen satisfaction with NYC 311 service requests based on:
    - **Agency Name**: The NYC agency handling the complaint
    - **Complaint Type**: Category of the service request  
    - **Borough**: NYC borough where the complaint originated
    - **Survey Year & Month**: When the satisfaction survey was conducted
    - **Cluster**: Complaint clustering based on patterns
    - **Sentiment Score**: Sentiment analysis of feedback text
    
    The model uses Random Forest classification and achieves:
    - **Recall**: 98.3%
    - **F1 Score**: 92.6% 
    - **AUC Score**: 97.1%
    """)