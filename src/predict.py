import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "../models/rf_v3.pkl")
feature_path = os.path.join(BASE_DIR, "../models/feature_list.pkl")

model = joblib.load(model_path)
feature_cols = joblib.load(feature_path)

def predict(input_df):
    X = input_df[feature_cols]
    preds = model.predict(X)
    return preds

if __name__ == "__main__":
    print("Predictive Aircraft Engine Maintenance — Inference Mode")

    sample_path = os.path.join(BASE_DIR, "../data/features/train_features_v3.csv")

    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        sample = df.sample(5, random_state=42)
        predictions = predict(sample)

        result = sample[['unit', 'cycle']].copy()
        result['predicted_RUL'] = predictions

        print(result)
    else:
        print("Sample file not found. Provide input dataframe to predict().")
