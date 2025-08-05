import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, roc_auc_score
import joblib

class SatisfactionPipeline:
    def __init__(self):
        self.pipeline = None
        self.label_encoders = {}
        self.feature_names = ['Agency Name', 'Complaint Type', 'Borough', 
                             'Survey Year', 'Survey Month', 'Cluster', 'Sentiment Score']
        
    def create_target(self, df):
        """Create binary satisfaction target"""
        df['Satisfied'] = df['Satisfaction Response'].apply(
            lambda x: 1 if x in ['Strongly Agree', 'Agree'] else 0
        )
        return df
    
    def preprocess_features(self, X):
        """Encode categorical features"""
        X_processed = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X_processed[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        return X_processed
    
    def build_pipeline(self):
        """Build the complete pipeline"""
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        return self.pipeline
    
    def fit(self, df):
        """Fit the pipeline on training data"""
        # Create target
        df = self.create_target(df)
        
        # Prepare features
        X = df[self.feature_names].copy()
        y = df['Satisfied']
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build and fit pipeline
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        print(f"Recall: {recall_score(y_test, y_pred):.3f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
        print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Feature importance
        feature_importance = pd.Series(
            self.pipeline.named_steps['classifier'].feature_importances_,
            index=self.feature_names
        )
        print("\nTop 5 Important Features:")
        print(feature_importance.nlargest(5))
        
        return self
    
    def predict(self, X):
        """Make predictions on new data"""
        X_processed = self.preprocess_features(X[self.feature_names])
        return self.pipeline.predict(X_processed)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_processed = self.preprocess_features(X[self.feature_names])
        return self.pipeline.predict_proba(X_processed)
    
    def save_model(self, filepath):
        """Save the complete pipeline"""
        model_data = {
            'pipeline': self.pipeline,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved pipeline"""
        model_data = joblib.load(filepath)
        instance = cls()
        instance.pipeline = model_data['pipeline']
        instance.label_encoders = model_data['label_encoders']
        instance.feature_names = model_data['feature_names']
        return instance

    
    
    
    
    
    
    
    