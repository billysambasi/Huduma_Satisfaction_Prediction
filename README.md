# NYC 311 Satisfaction Predictor

A machine learning application that predicts citizen satisfaction with NYC 311 service requests based on agency, complaint type, location, and timing data.

## Overview

This project analyzes NYC 311 service request data to predict citizen satisfaction levels. It includes data preprocessing, exploratory data analysis, machine learning model development, and deployment through both Streamlit and FastAPI interfaces.

## Features

- **Predictive Model**: Machine learning model trained on NYC 311 satisfaction survey data
- **Interactive Web App**: Streamlit interface for easy predictions
- **REST API**: FastAPI backend for programmatic access
- **Smart Filtering**: Dynamic dropdown menus that filter options based on selections
- **Real-time Predictions**: Get satisfaction probability scores instantly

## Project Structure

```
Phase5_Capstone/
├── app.py                              # Streamlit web application
├── fastapi_app.py                      # FastAPI REST API
├── data_preparation.ipynb              # Data cleaning and preprocessing
├── EDA.ipynb                          # Exploratory data analysis
├── modelling.ipynb                    # Model development and training
├── deployment.ipynb                   # Deployment preparation
├── satisfaction_model.pkl             # Trained ML model
├── encodings.pkl                      # Label encodings for categorical variables
├── agency_complaint_mapping.pkl       # Agency-complaint type mappings
├── complaint_descriptor_mapping.pkl   # Complaint-descriptor mappings
├── 311_Resolution_Satisfaction_Survey.csv  # Original dataset
├── cleaned.csv                        # Processed dataset
├── eda_incl.csv                      # EDA dataset
└── LICENSE                           # MIT License
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Phase5_Capstone
```

2. Install required dependencies:
```bash
pip install streamlit pandas joblib fastapi uvicorn pydantic
```

## Usage

### Streamlit Web Application

Run the interactive web application:
```bash
streamlit run app.py
```

The app provides:
- Agency selection dropdown
- Filtered complaint types based on selected agency
- Filtered descriptors based on selected complaint type
- Borough selection
- Year and month inputs
- Real-time satisfaction probability prediction

### FastAPI REST API

Start the API server:
```bash
uvicorn fastapi_app:app --reload
```

#### API Endpoints

- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `GET /options` - Get available options for dropdowns
- `POST /predict` - Make satisfaction predictions
- `POST /reload-model` - Reload model without restarting server

#### Example API Usage

```python
import requests

# Get prediction
response = requests.post("http://localhost:8000/predict", json={
    "agency_name": "Department of Transportation",
    "complaint_type": "Street Condition",
    "descriptor": "Pothole",
    "borough": "MANHATTAN",
    "year": 2024,
    "month": 6
})

print(response.json())
```

## Model Information

The satisfaction prediction model uses the following features:
- **Agency Name**: The NYC agency handling the complaint
- **Complaint Type**: Category of the service request
- **Descriptor**: Specific description of the issue
- **Borough**: NYC borough where the complaint originated
- **Survey Year**: Year of the satisfaction survey
- **Survey Month**: Month of the satisfaction survey

The model outputs a probability score (0-1) indicating the likelihood of citizen satisfaction with the service resolution.

## Data Pipeline

1. **Data Preparation** (`data_preparation.ipynb`): 
   - Data cleaning and preprocessing
   - Feature engineering
   - Handling missing values

2. **Exploratory Data Analysis** (`EDA.ipynb`):
   - Statistical analysis
   - Data visualization
   - Pattern identification

3. **Modeling** (`modelling.ipynb`):
   - Model selection and training
   - Hyperparameter tuning
   - Performance evaluation

4. **Deployment** (`deployment.ipynb`):
   - Model serialization
   - Deployment preparation

## Development

### Jupyter Notebooks

The project includes several Jupyter notebooks for different stages of development:

- **data_preparation.ipynb**: Data cleaning and preprocessing pipeline
- **EDA.ipynb**: Comprehensive exploratory data analysis
- **modelling.ipynb**: Model development, training, and evaluation
- **deployment.ipynb**: Deployment setup and testing

### Model Retraining

To retrain the model with new data:

1. Update the dataset files
2. Run the notebooks in sequence: data_preparation → EDA → modelling
3. Use the API's `/reload-model` endpoint to update the deployed model without restarting

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NYC Open Data for providing the 311 service request dataset
- Moringa School for project guidance and support

## Contact

For questions or support, please open an issue in the repository.