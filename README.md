# Concrete Chloride Corrosion XGBoost Model Prediction

Machine learning model for predicting corrosion rates in reinforced concrete structures using XGBoost with segmented modeling approach.

## 🌐 Live Demo

**Try the interactive app:** [chloride-corrosion-prediction.streamlit.app](https://chloride-corrosion-prediction.streamlit.app)

## Overview

This project implements a segmented XGBoost regression model to predict corrosion rates based on environmental and material properties. The model uses a dual-segment approach (low vs. high corrosion rates) with virtual sample generation to improve prediction accuracy.

## Features

- **Segmented Modeling**: Separate models for low (<0.15) and high (≥0.15) corrosion rates
- **Virtual Sample Generation**: Synthetic data augmentation for rare samples
- **Feature Engineering**: Interaction features (Humidity×Temperature, Cement×Cover)
- **Comprehensive Evaluation**: Multiple metrics including R², RMSE, MAE, and percentage error analysis

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
chloride_new/
├── data/                          # Data files
│   ├── Corrosion_Dataset_processed.csv
│   ├── predictions_training.csv
│   ├── predictions_testing.csv
│   └── sensitivity_*.csv
├── models/                        # Trained models
│   ├── model_low.pkl
│   ├── model_high.pkl
│   └── imputer.pkl
├── visualizations/                # Generated plots
│   ├── predictions_plot.png
│   ├── error_distribution.png
│   └── sensitivity_*.png
├── scripts/                       # Python scripts
│   ├── train_model.py
│   ├── predict.py
│   ├── app.py
│   └── sensitivity_analysis*.py
├── notebooks/                     # Jupyter notebooks
│   └── Modeling (5).ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The model expects a CSV file named `Corrosion_Dataset_processed.csv` in the `data/` folder with the following features:

- Cover_Thickness_mm
- Reinforcement_Diameter_mm
- Water_Cement_Ratio
- Temperature_K
- Relative_Humidity_pct
- Chloride_Ion_Content_kgm3
- Time_Years
- Corrosion_Rate_uAcm2 (target variable)

## Usage

### Training the Model

1. Ensure the dataset file `Corrosion_Dataset_processed.csv` is in the `data/` folder
2. Navigate to the scripts folder and run the training script:

```bash
cd scripts
python train_model.py
```

This will:
- Load and preprocess the data from `data/` folder
- Generate virtual samples (low and high rates)
- Train segmented XGBoost models
- Evaluate performance
- Generate visualizations (saved to `visualizations/`)
- Save trained models (saved to `models/`)

### Making Predictions

**Option 1: Live Web App (Recommended)**
🌐 **[Try the live app here](https://chloride-corrosion-prediction.streamlit.app)**

**Option 2: Local Web Interface**
```bash
cd scripts
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**Option 3: Command Line**
```bash
cd scripts
python predict.py
```

Or use programmatically:

```python
from predict import predict_from_dict

input_data = {
    "Cover_Thickness_mm": 50,
    "Reinforcement_Diameter_mm": 16,
    "Water_Cement_Ratio": 0.45,
    "Temperature_K": 298,
    "Relative_Humidity_pct": 75,
    "Chloride_Ion_Content_kgm3": 2.5,
    "Time_Years": 10
}

result = predict_from_dict(input_data)
print(f"Predicted corrosion rate: {result['final_prediction']:.3f} µA/cm²")
```

## Model Performance

- **R² Score**: 0.901
- **RMSE**: 0.0014
- **MAE**: 0.0236
- **Samples with ≤10% Error**: 46.94%
- **Samples with ≤5% Error**: 32.65%

## Feature Importance Analysis

The model identifies the following key factors affecting corrosion rates:

| Rank | Factor | Importance | Description |
|------|--------|------------|-------------|
| 1 | Relative_Humidity_pct | 36.23% | Critical environmental factor affecting moisture content |
| 2 | Humidity_Temp | 16.67% | Interaction between humidity and temperature |
| 3 | Temperature_K | 15.10% | Environmental factor affecting reaction kinetics |
| 4 | Chloride_Ion_Content_kgm3 | 14.68% | Primary corrosion agent concentration |
| 5 | Water_Cement_Ratio | 6.60% | Controls concrete permeability |
| 6 | Time_Years | 3.31% | Exposure duration effect |
| 7 | Cover_Thickness_mm | 3.21% | Physical barrier protection |
| 8 | Reinforcement_Diameter_mm | 1.96% | Steel reinforcement size |
| 9 | Cement_Cover | 2.24% | Interaction between cement ratio and cover |



