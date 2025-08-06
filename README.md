# GoK Huduma Satisfaction Predictor

A comprehensive machine learning application that predicts citizen satisfaction with NYC 311 service requests using advanced data science techniques including sentiment analysis, clustering and ensemble modeling.

## Overview

This project analyzes NYC 311 Resolution Satisfaction Survey data to predict citizen satisfaction levels with government service delivery. The system incorporates sophisticated data preprocessing, exploratory data analysis, feature engineering with NLP techniques, machine learning model development, and deployment through both interactive web interfaces and REST APIs.

## Key Features

- **Advanced ML Pipeline**: Custom satisfaction prediction pipeline with Random Forest classifier
- **NLP Integration**: Sentiment analysis and text clustering for feedback processing
- **Interactive Web App**: Streamlit interface with real-time predictions
- **REST API**: FastAPI backend for programmatic access and integration
- **Comprehensive Analytics**: Full data science workflow from EDA to deployment
- **High Performance**: Achieves 98.3% recall, 92.6% F1-score, and 97.1% AUC

## Project Architecture

```
Phase5_Capstone/
├── 311_Resolution_Satisfaction_Survey.csv  # Original dataset (364,689 records)
├── cleaned.csv                            # Processed dataset after cleaning
├── eda_incl.csv                          # Enhanced dataset with NLP features
├── satisfaction_model.pkl                 # Trained ML model with encoders
├── 
├── Core Application Files:
├── streamlit.py                          # Streamlit web application
├── app_api.py                           # FastAPI REST API server
├── satisfaction_pipeline.py             # Custom ML pipeline class
├── data_processor.py                    # Data loading and cleaning utilities
├── data_analysis.py                     # EDA and visualization classes
├── 
├── Analysis Notebooks:
├── data_preparation.ipynb               # Data cleaning and preprocessing
├── EDA.ipynb                           # Exploratory data analysis
├── modelling.ipynb                     # Model development and evaluation
├── deployment.ipynb                    # Deployment preparation
├── 
├── Configuration:
├── .gitignore                          # Git ignore rules
└── LICENSE                             # MIT License
```

## Technical Implementation

### Data Processing Pipeline

**1. Data Loading & Cleaning (`data_processor.py`)**
- Handles 364,689 survey responses across 11 features
- Intelligent missing value imputation using agency mappings
- Conditional filling of dissatisfaction reasons based on satisfaction levels
- Placeholder assignment for incomplete data fields

**2. Advanced Feature Engineering**
- **Sentiment Analysis**: VADER sentiment scoring on combined feedback text
- **Text Clustering**: K-means clustering on dissatisfaction reasons using TF-IDF
- **Categorical Encoding**: Label encoding for agency names, complaint types, and boroughs
- **Temporal Features**: Survey year and month for seasonal pattern detection

**3. Exploratory Data Analysis (`data_analysis.py`)**
- Comprehensive statistical analysis across 19 agencies and 206 complaint types
- Satisfaction distribution analysis by borough, agency, and complaint type
- Temporal trend analysis and seasonal pattern identification
- Interactive visualizations using matplotlib and seaborn

### Machine Learning Pipeline

**Custom SatisfactionPipeline Class (`satisfaction_pipeline.py`)**
```python
class SatisfactionPipeline:
    - Binary target creation (Satisfied: Strongly Agree/Agree vs Others)
    - Automated categorical encoding with label encoders
    - Random Forest classifier with 100 estimators
    - Built-in model persistence and loading capabilities
    - Feature importance analysis and performance metrics
```

**Model Performance:**
- **Recall**: 98.3% (excellent at identifying satisfied citizens)
- **F1 Score**: 92.6% (balanced precision and recall)
- **AUC Score**: 97.1% (strong discriminative ability)
- **Top Features**: Cluster (90.7%), Complaint Type (3.5%), Survey Month (1.8%)

**Model Comparison:**
| Metric | Random Forest | XGBoost | Selected |
|--------|---------------|---------|----------|
| F1 Score | 0.926 | 0.932 | ✓ Random Forest |
| Recall | 0.983 | 0.999 | (Better interpretability) |
| AUC Score | 0.971 | 0.963 | |

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install streamlit pandas numpy scikit-learn joblib fastapi uvicorn pydantic
pip install matplotlib seaborn nltk textblob wordcloud
pip install xgboost shap yellowbrick  # For advanced analysis
```

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd Phase5_Capstone

# Run Streamlit app
streamlit run streamlit.py

# Or run FastAPI server
uvicorn app_api:app --reload
```

## Application Interfaces

### Streamlit Web Application

**Features:**
- Interactive dropdown menus for all input features
- Real-time satisfaction probability prediction
- Visual probability indicators and interpretation
- Model performance metrics display
- User-friendly interface for non-technical users

**Usage:**
```bash
streamlit run streamlit.py
# Access at http://localhost:8501
```

### FastAPI REST API

**Endpoints:**
- `GET /` - API status and model information
- `GET /health` - Health check endpoint
- `POST /predict` - Satisfaction prediction endpoint
- `POST /reload-model` - Hot model reloading

**Prediction Request Format:**
```json
{
    "agency_name": "Department of Transportation",
    "complaint_type": "Street Condition", 
    "borough": "MANHATTAN",
    "year": 2024,
    "month": 6,
    "cluster": 2,
    "sentiment_score": 0.1
}
```

**Response Format:**
```json
{
    "satisfaction_probability": 0.8542,
    "prediction": 1,
    "prediction_label": "Satisfied",
    "model_loaded_at": "2024-01-15T10:30:00"
}
```

## Data Science Workflow

### 1. Data Understanding & Preparation

**Dataset Characteristics:**
- **Size**: 364,689 survey responses
- **Features**: 11 original features + 4 engineered features
- **Target**: Binary satisfaction (Strongly Agree/Agree vs Others)
- **Agencies**: 19 NYC government agencies
- **Complaint Types**: 206 different service request categories
- **Time Range**: 2022-2025 survey responses

**Data Quality Issues Addressed:**
- Missing agency names (94 records) - filled using acronym mappings
- Missing descriptors (12,975 records) - filled with "Not Provided"
- Missing dissatisfaction reasons (140,077 records) - conditionally filled
- Borough information gaps (438 records) - filled with "Unknown Borough"

### 2. Feature Engineering & NLP

**Text Processing Pipeline:**
```python
# Sentiment Analysis
from textblob import TextBlob
sentiment_scores = df['Combined_Feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Text Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
clusters = KMeans(n_clusters=5).fit_predict(tfidf_matrix)
```

**Engineered Features:**
- **Sentiment Score**: Polarity score (-1 to 1) from combined feedback text
- **Cluster**: K-means clusters (0-4) based on dissatisfaction patterns
- **Combined_Feedback**: Concatenated justified dissatisfaction and reason text
- **Sentiment Label**: Categorical sentiment (positive/neutral/negative)

### 3. Model Development & Evaluation

**Training Process:**
1. Binary target creation from 5-point satisfaction scale
2. Categorical feature encoding using LabelEncoder
3. 80/20 train-test split with stratification
4. Random Forest training with 100 estimators
5. Performance evaluation using multiple metrics

**Feature Importance Analysis:**
```
Cluster (90.7%): Dominant predictor based on dissatisfaction patterns
Complaint Type (3.5%): Service category influence
Survey Month (1.8%): Seasonal satisfaction variations
Sentiment Score (1.5%): Text sentiment impact
Borough (1.0%): Geographic satisfaction differences
```

## Advanced Analytics

### Satisfaction Analysis by Dimensions

**By Agency Performance:**
- NYPD: Highest volume (165,140 requests) with mixed satisfaction
- DOB: Building-related complaints with moderate satisfaction
- DSNY: Sanitation services with seasonal variations

**By Complaint Categories:**
- Illegal Parking: Most common (81,527 requests)
- Heat/Hot Water: High dissatisfaction in winter months
- Noise Complaints: Consistent moderate satisfaction

**By Geographic Distribution:**
- Brooklyn: Highest volume (129,662 requests)
- Manhattan: Higher satisfaction rates
- Staten Island: Lowest volume but higher satisfaction

### Temporal Patterns
- **Seasonal Trends**: Higher dissatisfaction in winter months
- **Year-over-Year**: Gradual improvement in satisfaction rates
- **Monthly Variations**: Peak complaint volumes in summer

## Model Interpretability

**SHAP Analysis Integration:**
```python
import shap
explainer = shap.TreeExplainer(model.pipeline.named_steps['classifier'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Business Insights:**
1. **Cluster Analysis**: Dissatisfaction patterns are the strongest predictor
2. **Service Type Impact**: Certain complaint types consistently underperform
3. **Temporal Effects**: Satisfaction varies by season and survey timing
4. **Geographic Disparities**: Borough-level satisfaction differences

## Deployment & Production

### Model Persistence
```python
# Save complete pipeline with encoders
model_data = {
    'pipeline': self.pipeline,
    'label_encoders': self.label_encoders,
    'feature_names': self.feature_names
}
joblib.dump(model_data, 'satisfaction_model.pkl')
```

### API Integration
- **Hot Reloading**: Update models without server restart
- **Error Handling**: Comprehensive exception management
- **Input Validation**: Pydantic models for request validation
- **Logging**: Request/response logging for monitoring

### Performance Monitoring
- **Model Metrics**: Continuous performance tracking
- **Data Drift**: Feature distribution monitoring
- **Prediction Confidence**: Probability threshold analysis

## Development Workflow

### Notebook-Driven Development
1. **data_preparation.ipynb**: Data cleaning and initial preprocessing
2. **EDA.ipynb**: Comprehensive exploratory analysis with visualizations
3. **modelling.ipynb**: Model development, comparison, and evaluation
4. **deployment.ipynb**: Production deployment preparation

### Code Organization
- **Object-Oriented Design**: Modular classes for different functionalities
- **Pipeline Architecture**: Reusable ML pipeline with sklearn compatibility
- **Separation of Concerns**: Clear separation between data, model, and API layers

## Future Enhancements

### Technical Improvements
- **Deep Learning**: LSTM/BERT models for text analysis
- **Ensemble Methods**: Stacking multiple algorithms
- **Real-time Learning**: Online learning capabilities
- **A/B Testing**: Model comparison framework

### Business Applications
- **Proactive Interventions**: Early warning system for dissatisfaction
- **Resource Allocation**: Optimize agency resource distribution
- **Service Quality Metrics**: KPI dashboard for agencies
- **Citizen Feedback Loop**: Automated response system

## Contributing

### Development Setup
```bash
# Fork and clone repository
git clone https://github.com/billysambasi/Phase5_Capstone.git
cd Phase5_Capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Contribution Guidelines
1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Update docstrings and README
3. **Testing**: Add unit tests for new features
4. **Performance**: Benchmark model changes
5. **Security**: Validate input sanitization

## License & Acknowledgments

**License**: MIT License - see [LICENSE](LICENSE) file

**Data Source**: [NYC Open Data - 311 Resolution Satisfaction Survey](https://catalog.data.gov/dataset/311-resolution-satisfaction-survey-e4003)

**Technologies Used**:
- **ML/DS**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **NLP**: NLTK, TextBlob, TF-IDF vectorization
- **Web**: Streamlit, FastAPI, Uvicorn
- **Deployment**: Joblib model persistence, Pydantic validation

---

*This project demonstrates end-to-end machine learning implementation for government service satisfaction prediction, incorporating modern data science best practices and production-ready deployment strategies.*