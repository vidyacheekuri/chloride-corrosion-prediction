"""
Chloride Corrosion XGBoost Model - Prediction Script
Make predictions using trained segmented models
"""

import pandas as pd
import numpy as np
import joblib


def load_models():
    """Load trained models and imputer"""
    try:
        model_low = joblib.load("../models/model_low.pkl")
        model_high = joblib.load("../models/model_high.pkl")
        imputer = joblib.load("../models/imputer.pkl")
        return model_low, model_high, imputer
    except FileNotFoundError:
        raise FileNotFoundError(
            "Model files not found. Please run train_model.py first to train the models."
        )


def prepare_input_features(input_dict, imputer, feature_columns):
    """Prepare input features with interaction terms"""
    # Create DataFrame from input
    df_input = pd.DataFrame([input_dict])
    
    # Add interaction features
    df_input["Cement_Cover"] = df_input["Cover_Thickness_mm"] * df_input["Water_Cement_Ratio"]
    df_input["Humidity_Temp"] = df_input["Relative_Humidity_pct"] * df_input["Temperature_K"]
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Reorder columns to match training
    df_input = df_input[feature_columns]
    
    # Apply imputation
    X_input = pd.DataFrame(
        imputer.transform(df_input),
        columns=feature_columns
    )
    
    return X_input


def predict_segmented(X_new, model_low, model_high, threshold=0.15):
    """Make prediction using segmented models"""
    pred_low = model_low.predict(X_new)
    pred_high = model_high.predict(X_new)
    
    # Choose low-model prediction if it's below threshold, otherwise high-model
    final_pred = np.where(pred_low <= threshold, pred_low, pred_high)
    
    return final_pred[0], pred_low[0], pred_high[0]


def predict_from_user_input():
    """Interactive prediction from user input"""
    print("="*60)
    print("🔬 Chloride Corrosion Rate Prediction")
    print("="*60)
    print("\nPlease enter the following parameters:\n")
    
    try:
        cover_mm = float(input("Cover thickness (mm): "))
        diameter_mm = float(input("Reinforcement diameter (mm): "))
        wcr = float(input("Water-cement ratio (e.g. 0.40): "))
        temp_K = float(input("Temperature (K): "))
        rh_pct = float(input("Relative humidity (%): "))
        chloride_kgm3 = float(input("Chloride ion content (kg/m³): "))
        time_years = float(input("Exposure time (years): "))
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
        return
    
    input_dict = {
        "Cover_Thickness_mm": cover_mm,
        "Reinforcement_Diameter_mm": diameter_mm,
        "Water_Cement_Ratio": wcr,
        "Temperature_K": temp_K,
        "Relative_Humidity_pct": rh_pct,
        "Chloride_Ion_Content_kgm3": chloride_kgm3,
        "Time_Years": time_years
    }
    
    # Load models
    model_low, model_high, imputer = load_models()
    
    # Get feature columns from a saved model attribute or define them
    feature_columns = [
        'Cover_Thickness_mm', 'Reinforcement_Diameter_mm', 'Water_Cement_Ratio',
        'Temperature_K', 'Relative_Humidity_pct', 'Chloride_Ion_Content_kgm3',
        'Time_Years', 'Humidity_Temp', 'Cement_Cover'
    ]
    
    # Prepare features
    X_input = prepare_input_features(input_dict, imputer, feature_columns)
    
    # Make prediction
    final_pred, pred_low, pred_high = predict_segmented(X_input, model_low, model_high)
    
    # Display results
    print("\n" + "="*60)
    print("📊 Prediction Results")
    print("="*60)
    print(f"Low-rate segment prediction:  {pred_low:.3f} µA/cm²")
    print(f"High-rate segment prediction: {pred_high:.3f} µA/cm²")
    print(f"\n🎯 Final segmented prediction: {final_pred:.3f} µA/cm²")
    print("="*60)


def predict_from_dict(input_dict):
    """Make prediction from a dictionary of input parameters"""
    model_low, model_high, imputer = load_models()
    
    feature_columns = [
        'Cover_Thickness_mm', 'Reinforcement_Diameter_mm', 'Water_Cement_Ratio',
        'Temperature_K', 'Relative_Humidity_pct', 'Chloride_Ion_Content_kgm3',
        'Time_Years', 'Humidity_Temp', 'Cement_Cover'
    ]
    
    X_input = prepare_input_features(input_dict, imputer, feature_columns)
    final_pred, pred_low, pred_high = predict_segmented(X_input, model_low, model_high)
    
    return {
        'final_prediction': final_pred,
        'low_segment_prediction': pred_low,
        'high_segment_prediction': pred_high
    }


def main():
    """Main prediction interface"""
    predict_from_user_input()


if __name__ == "__main__":
    main()

