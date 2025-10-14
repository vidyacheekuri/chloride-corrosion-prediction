"""
Chloride Corrosion Prediction - Minimal Web Interface
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Corrosion Predictor", page_icon="🔬", layout="centered")

@st.cache_resource
def load_models():
    """Load the trained models and imputer"""
    # Try different paths for local vs deployed
    paths_to_try = [
        "../models/model_low.pkl",      # Local path
        "models/model_low.pkl",         # Deployed path
        "./models/model_low.pkl"        # Alternative deployed path
    ]
    
    for base_path in paths_to_try:
        try:
            model_low = joblib.load(base_path)
            model_high = joblib.load(base_path.replace("model_low", "model_high"))
            imputer = joblib.load(base_path.replace("model_low", "imputer"))
            return model_low, model_high, imputer
        except FileNotFoundError:
            continue
    
    st.error("⚠️ Model files not found! Tried paths: " + ", ".join(paths_to_try))
    st.stop()

def predict_corrosion(input_data, model_low, model_high, imputer, threshold=0.15):
    """Make prediction using the segmented models"""
    input_data["Cement_Cover"] = input_data["Cover_Thickness_mm"] * input_data["Water_Cement_Ratio"]
    input_data["Humidity_Temp"] = input_data["Relative_Humidity_pct"] * input_data["Temperature_K"]
    
    feature_columns = [
        'Cover_Thickness_mm', 'Reinforcement_Diameter_mm', 'Water_Cement_Ratio',
        'Temperature_K', 'Relative_Humidity_pct', 'Chloride_Ion_Content_kgm3',
        'Time_Years', 'Humidity_Temp', 'Cement_Cover'
    ]
    
    X_input = pd.DataFrame(imputer.transform(input_data[feature_columns]), columns=feature_columns)
    pred_low = model_low.predict(X_input)[0]
    pred_high = model_high.predict(X_input)[0]
    final_pred = pred_low if pred_low <= threshold else pred_high
    
    return final_pred

def main():
    model_low, model_high, imputer = load_models()
    
    st.title("Concrete ChlorideCorrosion Rate Predictor")
    st.caption("XGBoost-based model for reinforced concrete structures")
    
    # Input fields in 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        cover_thickness = st.number_input("Cover Thickness (mm)", 0.0, 200.0, 50.0)
        reinforcement_diameter = st.number_input("Reinforcement Diameter (mm)", 0.0, 50.0, 16.0)
        water_cement_ratio = st.number_input("Water-Cement Ratio", 0.0, 1.0, 0.45, 0.01)
        temperature = st.number_input("Temperature (K)", 250.0, 350.0, 298.0)
    
    with col2:
        humidity = st.number_input("Relative Humidity (%)", 0.0, 100.0, 75.0)
        chloride_content = st.number_input("Chloride Content (kg/m³)", 0.0, 15.0, 2.5, 0.1)
        exposure_time = st.number_input("Exposure Time (years)", 0.0, 100.0, 10.0)
    
    if st.button("Predict", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Cover_Thickness_mm": cover_thickness,
            "Reinforcement_Diameter_mm": reinforcement_diameter,
            "Water_Cement_Ratio": water_cement_ratio,
            "Temperature_K": temperature,
            "Relative_Humidity_pct": humidity,
            "Chloride_Ion_Content_kgm3": chloride_content,
            "Time_Years": exposure_time
        }])
        
        with st.spinner("Calculating..."):
            prediction = predict_corrosion(input_data, model_low, model_high, imputer)
        
        st.divider()
        
        # Display result
        st.markdown(f"""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;'>
            <h1 style='color: white; margin: 0; font-size: 3.5em;'>{prediction:.3f}</h1>
            <p style='color: white; font-size: 1.5em; margin: 10px 0 0 0;'>µA/cm²</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple severity indicator
        if prediction < 0.15:
            st.success("✅ Low corrosion risk")
        elif prediction < 0.5:
            st.warning("⚠️ Moderate corrosion risk")
        else:
            st.error("🚨 High corrosion risk")

if __name__ == "__main__":
    main()
