"""
Chloride Corrosion Prediction - Web Interface
Interactive Streamlit app for corrosion rate predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Corrosion Rate Predictor",
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load the trained models and imputer"""
    try:
        model_low = joblib.load("../models/model_low.pkl")
        model_high = joblib.load("../models/model_high.pkl")
        imputer = joblib.load("../models/imputer.pkl")
        return model_low, model_high, imputer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run train_model.py first.")
        st.stop()

def predict_corrosion(input_data, model_low, model_high, imputer, threshold=0.15):
    """Make prediction using the segmented models"""
    # Create interaction features
    input_data["Cement_Cover"] = input_data["Cover_Thickness_mm"] * input_data["Water_Cement_Ratio"]
    input_data["Humidity_Temp"] = input_data["Relative_Humidity_pct"] * input_data["Temperature_K"]
    
    # Define feature columns in correct order
    feature_columns = [
        'Cover_Thickness_mm', 'Reinforcement_Diameter_mm', 'Water_Cement_Ratio',
        'Temperature_K', 'Relative_Humidity_pct', 'Chloride_Ion_Content_kgm3',
        'Time_Years', 'Humidity_Temp', 'Cement_Cover'
    ]
    
    # Prepare features
    X_input = pd.DataFrame(
        imputer.transform(input_data[feature_columns]),
        columns=feature_columns
    )
    
    # Get predictions from both models
    pred_low = model_low.predict(X_input)[0]
    pred_high = model_high.predict(X_input)[0]
    
    # Choose final prediction based on threshold
    final_pred = pred_low if pred_low <= threshold else pred_high
    
    return final_pred, pred_low, pred_high

# Main app
def main():
    # Load models
    model_low, model_high, imputer = load_models()
    
    # Header
    st.title("🔬 Chloride Corrosion Rate Predictor")
    st.markdown("### XGBoost-based Segmented Model for Corrosion Prediction")
    st.markdown("---")
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts corrosion rates in reinforced concrete structures 
        using a segmented XGBoost model.
        
        **Model Performance:**
        - R² Score: 0.902
        - MAE: 0.023
        - 53% within 10% error
        """)
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.markdown("""
        - **Low-rate model**: < 0.15 µA/cm²
        - **High-rate model**: ≥ 0.15 µA/cm²
        """)
    
    # Input section
    st.header("📝 Input Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Structural Properties")
        cover_thickness = st.number_input(
            "Cover Thickness (mm)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=1.0,
            help="Concrete cover thickness protecting the reinforcement"
        )
        
        reinforcement_diameter = st.number_input(
            "Reinforcement Diameter (mm)",
            min_value=0.0,
            max_value=50.0,
            value=16.0,
            step=1.0,
            help="Diameter of the steel reinforcement bar"
        )
        
        water_cement_ratio = st.number_input(
            "Water-Cement Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            help="Ratio of water to cement in the concrete mix"
        )
    
    with col2:
        st.subheader("Environmental Conditions")
        temperature = st.number_input(
            "Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=298.0,
            step=1.0,
            help="Ambient temperature in Kelvin"
        )
        
        humidity = st.number_input(
            "Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
            help="Relative humidity percentage"
        )
        
        chloride_content = st.number_input(
            "Chloride Ion Content (kg/m³)",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Chloride ion concentration in concrete"
        )
    
    with col3:
        st.subheader("Time")
        exposure_time = st.number_input(
            "Exposure Time (years)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Duration of exposure to corrosive environment"
        )
        
        st.markdown("")
        st.markdown("")
        predict_button = st.button("🔍 Predict Corrosion Rate", type="primary", use_container_width=True)
    
    # Prediction section
    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame([{
            "Cover_Thickness_mm": cover_thickness,
            "Reinforcement_Diameter_mm": reinforcement_diameter,
            "Water_Cement_Ratio": water_cement_ratio,
            "Temperature_K": temperature,
            "Relative_Humidity_pct": humidity,
            "Chloride_Ion_Content_kgm3": chloride_content,
            "Time_Years": exposure_time
        }])
        
        # Make prediction
        with st.spinner("Calculating..."):
            final_pred, pred_low, pred_high = predict_corrosion(
                input_data, model_low, model_high, imputer
            )
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Main prediction - centered
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h1 style='color: #1f77b4; margin: 0;'>{final_pred:.3f}</h1>
                    <p style='font-size: 20px; margin: 10px 0 0 0;'>µA/cm²</p>
                    <p style='font-size: 14px; color: #666; margin: 5px 0 0 0;'>Predicted Corrosion Rate</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Interpretation
        st.markdown("---")
        st.subheader("📊 Interpretation")
        
        if final_pred < 0.05:
            severity = "Very Low"
            color = "🟢"
            message = "Minimal corrosion risk. The structure is in good condition."
        elif final_pred < 0.15:
            severity = "Low"
            color = "🟡"
            message = "Low corrosion rate. Regular monitoring recommended."
        elif final_pred < 0.5:
            severity = "Moderate"
            color = "🟠"
            message = "Moderate corrosion. Consider preventive measures."
        else:
            severity = "High"
            color = "🔴"
            message = "High corrosion rate. Immediate attention required."
        
        st.info(f"{color} **Severity Level:** {severity}\n\n{message}")
        
        # Additional info
        with st.expander("ℹ️ Understanding the Results"):
            st.markdown("""
            **Corrosion Rate Units:** µA/cm² (microamperes per square centimeter)
            
            **Model Approach:**
            - The model uses two specialized sub-models
            - Low-rate model: Optimized for rates < 0.15 µA/cm²
            - High-rate model: Optimized for rates ≥ 0.15 µA/cm²
            - Final prediction is automatically selected based on the threshold
            
            **Factors Affecting Corrosion:**
            - Higher chloride content → Higher corrosion
            - Higher humidity → Higher corrosion
            - Longer exposure time → Higher corrosion
            - Thicker concrete cover → Lower corrosion
            """)

if __name__ == "__main__":
    main()

