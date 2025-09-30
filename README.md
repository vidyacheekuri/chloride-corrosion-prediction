# Chloride Corrosion Prediction

A comprehensive machine learning pipeline for predicting chloride-induced corrosion rates in reinforced concrete structures using ensemble methods and sensitivity analysis.

## Overview

This project implements a complete machine learning workflow for chloride corrosion prediction, including:

- **Data Augmentation**: Safe augmentation techniques to increase training dataset size
- **Ensemble Modeling**: Multi-model approach using MLP and XGBoost regressors
- **Sensitivity Analysis**: Comprehensive input variable sensitivity analysis
- **Performance Evaluation**: Detailed model performance metrics and visualizations

## Features

- **Safe Data Augmentation**: Percentage-based noise augmentation that preserves data integrity
- **Ensemble Learning**: Voting regressor combining MLP and XGBoost models
- **Comprehensive Evaluation**: Multiple performance metrics and error analysis
- **Sensitivity Analysis**: Input variable perturbation analysis with visualizations
- **Professional Code Structure**: Modular, well-documented, and maintainable codebase

## Project Structure

```
chloride_corrosion_ml/
├── main.py                     # Main pipeline orchestrator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/                        # Source code modules
│   ├── data_augmentation.py    # Data augmentation module
│   ├── model_training.py       # Model training and evaluation
│   └── sensitivity_analysis.py # Sensitivity analysis module
├── data/                       # Data directory
│   ├── Chloride_dataset.csv    # Input dataset
│   ├── testing_data.csv        # Test dataset
│   ├── augmented_training_data_unscaled.csv  # Augmented training data
│   └── original_training_data.csv            # Original training data
└── results/                    # Results and outputs
    ├── chloride_corrosion_model.pkl          # Trained model
    ├── feature_scaler.pkl                    # Feature scaler
    ├── final_model_predictions.csv           # Prediction results
    ├── model_feature_importance.csv          # Feature importance
    ├── model_performance_summary.csv         # Performance metrics
    ├── feature_sensitivity_ranking.csv       # Sensitivity ranking
    ├── detailed_sensitivity_analysis.csv     # Detailed sensitivity results
    ├── sensitivity_summary_table.csv         # Sensitivity summary
    ├── sensitivity_ranking.png               # Sensitivity ranking plot
    ├── sensitivity_heatmap.png               # Sensitivity heatmap
    └── individual_sensitivities.png          # Individual sensitivity curves
```

## Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd chloride_corrosion_ml
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Place your input dataset as `data/Chloride_dataset.csv`
   - Place your test dataset as `data/testing_data.csv`
   - Ensure the CSV files have the required columns (see Data Format section)

## Usage

### Complete Pipeline

Run the entire pipeline with default settings:

```bash
python main.py --input-data data/Chloride_dataset.csv --test-data data/testing_data.csv
```

### Individual Steps

Run specific pipeline steps:

```bash
# Data augmentation only
python main.py --step augmentation --input-data data/Chloride_dataset.csv

# Model training only
python main.py --step training --input-data data/Chloride_dataset.csv --test-data data/testing_data.csv

# Sensitivity analysis only
python main.py --step sensitivity --input-data data/Chloride_dataset.csv --test-data data/testing_data.csv
```

### Advanced Options

```bash
# Skip data augmentation
python main.py --input-data data/Chloride_dataset.csv --test-data data/testing_data.csv --skip-augmentation

# Custom output directory
python main.py --input-data data/Chloride_dataset.csv --test-data data/testing_data.csv --output-dir custom_results

# Debug mode
python main.py --input-data data/Chloride_dataset.csv --test-data data/testing_data.csv --log-level DEBUG
```

## Data Format

### Input Dataset (Chloride_dataset.csv)

The input dataset should contain the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| Cover_Thickness_mm | Concrete cover thickness | mm |
| Reinforcement_Diameter_mm | Reinforcement bar diameter | mm |
| Water_Cement_Ratio | Water-to-cement ratio | - |
| Temperature_K | Temperature | Kelvin |
| Relative_Humidity_pct | Relative humidity | % |
| Chloride_Ion_Content_kgm3 | Chloride ion content | kg/m³ |
| Time_Years | Time exposure | years |
| Corrosion_Rate_uAcm2 | Corrosion rate (target variable) | μA/cm² |

### Test Dataset (testing_data.csv)

Same format as input dataset, used for model evaluation.

## Model Architecture

### Ensemble Model

The pipeline uses a voting regressor combining three models:

1. **Multi-layer Perceptron (MLP)**:
   - Architecture: 100-50-25 neurons
   - Activation: ReLU
   - Regularization: L2 (α=0.001)
   - Early stopping enabled

2. **XGBoost Model 1** (Primary):
   - 400 estimators
   - Max depth: 6
   - Learning rate: 0.08
   - Regularization: α=0.1, λ=0.1

3. **XGBoost Model 2** (Secondary):
   - 300 estimators
   - Max depth: 5
   - Learning rate: 0.1
   - Regularization: α=0.05, λ=0.05

**Ensemble Weights**: MLP (30%), XGBoost1 (40%), XGBoost2 (30%)

### Data Preprocessing

- **Feature Scaling**: RobustScaler (robust to outliers)
- **Target Transformation**: Log1p transformation for corrosion rates
- **Data Augmentation**: Safe percentage-based noise augmentation

## Performance Metrics

The model evaluation includes:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Error Rate Analysis**: Percentage of predictions within error thresholds (5%, 10%, 15%, 20%, 30%)
- **Feature Importance**: XGBoost-based feature importance ranking

## Sensitivity Analysis

The sensitivity analysis evaluates how input variable perturbations affect predictions:

- **Perturbation Levels**: -10%, -5%, +5%, +10%
- **Analysis Method**: One-at-a-time perturbation
- **Outputs**: 
  - Sensitivity ranking
  - Detailed impact analysis
  - Visualization plots
  - Practical recommendations

## Output Files

### Model Files
- `chloride_corrosion_model.pkl`: Trained ensemble model
- `feature_scaler.pkl`: Fitted feature scaler

### Results Files
- `final_model_predictions.csv`: Detailed predictions vs actual values
- `model_feature_importance.csv`: Feature importance rankings
- `model_performance_summary.csv`: Performance metrics summary

### Sensitivity Analysis Files
- `feature_sensitivity_ranking.csv`: Overall sensitivity rankings
- `detailed_sensitivity_analysis.csv`: Sample-level sensitivity results
- `sensitivity_summary_table.csv`: Summary table format
- `sensitivity_ranking.png`: Tornado diagram
- `sensitivity_heatmap.png`: Sensitivity heatmap
- `individual_sensitivities.png`: Individual feature sensitivity curves

## API Reference

### DataAugmenter Class

```python
from src.data_augmentation import DataAugmenter

augmenter = DataAugmenter(target_samples=800)
X_aug, y_aug = augmenter.augment_dataset('data/input.csv')
```

### ChlorideCorrosionPredictor Class

```python
from src.model_training import ChlorideCorrosionPredictor

predictor = ChlorideCorrosionPredictor()
results = predictor.train_and_evaluate('data/train.csv', 'data/test.csv')
```

### SensitivityAnalyzer Class

```python
from src.sensitivity_analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer()
sensitivity_df, detailed_df = analyzer.run_complete_analysis(
    'data/train.csv', 'data/test.csv'
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of a research assistantship. Please ensure proper attribution and academic integrity when using this code.

## Contact

For questions or issues, please contact the research team.

## Changelog

### Version 1.0.0
- Initial release
- Complete pipeline implementation
- Data augmentation module
- Model training and evaluation
- Sensitivity analysis module
- Professional documentation
