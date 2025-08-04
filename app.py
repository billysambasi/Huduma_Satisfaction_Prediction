# import streamlit as st
# import pandas as pd
# import joblib

# # Load model and encodings
# model = joblib.load('satisfaction_model.pkl')
# encodings = joblib.load('encodings.pkl')

# st.title('GoK Huduma Satisfaction Predictor')

# # Input fields
# agency = st.selectbox('Agency', list(encodings['Agency Name'].keys()))
# complaint_type = st.selectbox('Complaint Type', list(encodings['Complaint Type'].keys()))
# descriptor = st.selectbox('Descriptor', list(encodings['Descriptor'].keys()))
# borough = st.selectbox('Borough', list(encodings['Borough'].keys()))
# year = st.number_input('Year', min_value=2020, max_value=2025, value=2024)
# month = st.number_input('Month', min_value=1, max_value=12, value=1)

# if st.button('Predict Satisfaction'):
#     # Create input dataframe
#     input_data = pd.DataFrame({
#         'Agency Name': [encodings['Agency Name'][agency]],
#         'Complaint Type': [encodings['Complaint Type'][complaint_type]],
#         'Descriptor': [encodings['Descriptor'][descriptor]],
#         'Borough': [encodings['Borough'][borough]],
#         'Survey Year': [year],
#         'Survey Month': [month]
#     })
    
#     prediction = model.predict_proba(input_data)[0][1]
#     st.write(f'Satisfaction Probability: {prediction:.2%}')


# import streamlit as st
# import pandas as pd
# import joblib

# # Load model and encodings
# model = joblib.load('satisfaction_model.pkl')
# encodings = joblib.load('encodings.pkl')

# # Load the original data to create agency-complaint mapping
# @st.cache_data
# def load_agency_complaint_mapping():
#     df = pd.read_csv('cleaned.csv')
#     # Create mapping of agency to its complaint types
#     agency_complaints = df.groupby('Agency Name')['Complaint Type'].unique().to_dict()
#     return agency_complaints

# agency_complaint_map = load_agency_complaint_mapping()

# st.title('NYC 311 Satisfaction Predictor')

# # Agency dropdown
# agency = st.selectbox('Agency', list(encodings['Agency Name'].keys()))

# # Filtered complaint types based on selected agency
# available_complaints = agency_complaint_map.get(agency, [])
# complaint_type = st.selectbox('Complaint Type', available_complaints)

# # Similarly, you could filter descriptors by complaint type
# descriptor = st.selectbox('Descriptor', list(encodings['Descriptor'].keys()))
# borough = st.selectbox('Borough', list(encodings['Borough'].keys()))
# year = st.number_input('Year', min_value=2020, max_value=2025, value=2024)
# month = st.number_input('Month', min_value=1, max_value=12, value=1)

# if st.button('Predict Satisfaction'):
#     # Create input dataframe
#     input_data = pd.DataFrame({
#         'Agency Name': [encodings['Agency Name'][agency]],
#         'Complaint Type': [encodings['Complaint Type'][complaint_type]],
#         'Descriptor': [encodings['Descriptor'][descriptor]],
#         'Borough': [encodings['Borough'][borough]],
#         'Survey Year': [year],
#         'Survey Month': [month]
#     })
    
#     prediction = model.predict_proba(input_data)[0][1]
#     st.write(f'Satisfaction Probability: {prediction:.2%}')

import streamlit as st
import pandas as pd
import joblib

# Load model, encodings, and mappings
model = joblib.load('satisfaction_model.pkl')
encodings = joblib.load('encodings.pkl')
agency_complaint_map = joblib.load('agency_complaint_mapping.pkl')
complaint_descriptor_map = joblib.load('complaint_descriptor_mapping.pkl')

st.title('NYC 311 Satisfaction Predictor')

# Agency dropdown
agency = st.selectbox('Agency', list(encodings['Agency Name'].keys()))

# Filtered complaint types based on selected agency
available_complaints = agency_complaint_map.get(agency, [])
complaint_type = st.selectbox('Complaint Type', available_complaints)

# Filtered descriptors based on selected complaint type
available_descriptors = complaint_descriptor_map.get(complaint_type, [])
descriptor = st.selectbox('Descriptor', available_descriptors)

borough = st.selectbox('Borough', list(encodings['Borough'].keys()))
year = st.number_input('Year', min_value=2020, max_value=2025, value=2024)
month = st.number_input('Month', min_value=1, max_value=12, value=1)

if st.button('Predict Satisfaction'):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Agency Name': [encodings['Agency Name'][agency]],
        'Complaint Type': [encodings['Complaint Type'][complaint_type]],
        'Descriptor': [encodings['Descriptor'][descriptor]],
        'Borough': [encodings['Borough'][borough]],
        'Survey Year': [year],
        'Survey Month': [month]
    })
    
    prediction = model.predict_proba(input_data)[0][1]
    st.write(f'Satisfaction Probability: {prediction:.2%}')
