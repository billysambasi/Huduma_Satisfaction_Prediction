import pandas as pd
import requests
import json
from datetime import datetime
import time

def batch_predict_csv(csv_file, output_file="powerbi_predictions.csv"):
    """
    Read CSV, make predictions, save results for Power BI
    """
    # Read your CSV file
    df = pd.read_csv(csv_file)
    
    # API endpoint
    api_url = "http://localhost:8000/predict"
    
    results = []
    
    print(f"Processing {len(df)} rows...")
    
    for index, row in df.iterrows():
        try:
            # Prepare data for API (adjust column names as needed)
            prediction_data = {
                "agency_name": row.get("agency_name", "Unknown"),
                "complaint_type": row.get("complaint_type", "Unknown"),
                "borough": row.get("borough", "Unknown"),
                "year": int(row.get("year", 2024)),
                "month": int(row.get("month", 1)),
                "cluster": int(row.get("cluster", 0)),
                "sentiment_score": float(row.get("sentiment_score", 0.0))
            }
            
            # Make prediction
            response = requests.post(api_url, json=prediction_data, timeout=10)
            
            if response.status_code == 200:
                pred_result = response.json()
                
                # Combine original data with prediction
                result_row = {
                    "timestamp": datetime.now().isoformat(),
                    "agency_name": prediction_data["agency_name"],
                    "complaint_type": prediction_data["complaint_type"],
                    "borough": prediction_data["borough"],
                    "year": prediction_data["year"],
                    "month": prediction_data["month"],
                    "satisfaction_prediction": pred_result.get("prediction_label", "Unknown"),
                    "satisfaction_probability": pred_result.get("satisfaction_probability", 0.0),
                    "cluster": prediction_data["cluster"],
                    "sentiment_score": prediction_data["sentiment_score"]
                }
                
                results.append(result_row)
                print(f"Processed row {index + 1}/{len(df)}")
                
            else:
                print(f"API error for row {index + 1}: {response.status_code}")
                
        except Exception as e:
            print(f"Error processing row {index + 1}: {e}")
        
        # Small delay to avoid overwhelming API
        time.sleep(0.1)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(results)} predictions to {output_file}")
    return results_df

if __name__ == "__main__":
    # Replace with your CSV file name
    csv_file = "data/processed/cleaned.csv"
    batch_predict_csv(csv_file)