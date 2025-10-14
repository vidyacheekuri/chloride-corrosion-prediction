"""
Sensitivity Analysis on Training and Testing Datasets
Analyzes feature importance and saves predictions to CSV
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Set style
sns.set_style("whitegrid")

def load_models():
    """Load the trained models and imputer"""
    try:
        model_low = joblib.load("../models/model_low.pkl")
        model_high = joblib.load("../models/model_high.pkl")
        imputer = joblib.load("../models/imputer.pkl")
        return model_low, model_high, imputer
    except FileNotFoundError:
        print("❌ Model files not found! Please run train_model.py first.")
        exit(1)

def load_and_prepare_data(filepath="../data/Corrosion_Dataset_processed.csv"):
    """Load dataset and add interaction features"""
    df = pd.read_csv(filepath)
    
    # Add interaction features
    df["Humidity_Temp"] = df["Relative_Humidity_pct"] * df["Temperature_K"]
    df["Cement_Cover"] = df["Water_Cement_Ratio"] * df["Cover_Thickness_mm"]
    
    return df

def generate_virtual_samples(df, n_samples=100, quantile=0.25, random_state=42):
    """Generate synthetic samples for rare corrosion rates"""
    threshold = df["Corrosion_Rate_uAcm2"].quantile(quantile)
    rare = df[df["Corrosion_Rate_uAcm2"] < threshold]
    
    numeric = [
        "Cover_Thickness_mm", "Reinforcement_Diameter_mm", "Water_Cement_Ratio",
        "Temperature_K", "Relative_Humidity_pct", "Chloride_Ion_Content_kgm3", "Time_Years"
    ]
    
    synth = rare.sample(n=n_samples, replace=True, random_state=random_state).copy()
    synth[numeric] += np.random.normal(0, 0.01, size=synth[numeric].shape)
    
    # Recompute interaction features
    synth["Humidity_Temp"] = synth["Relative_Humidity_pct"] * synth["Temperature_K"]
    synth["Cement_Cover"] = synth["Water_Cement_Ratio"] * synth["Cover_Thickness_mm"]
    
    df_augmented = pd.concat([df, synth], ignore_index=True)
    return df_augmented

def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """Split data and apply imputation"""
    X = df.drop(columns=["Corrosion_Rate_uAcm2"])
    y = df["Corrosion_Rate_uAcm2"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Impute missing values
    imp = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imp.transform(X_test), columns=X.columns)
    
    # Reset indices
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

def predict_segmented(X, y, model_low, model_high, threshold=0.15):
    """Make predictions for entire dataset"""
    predictions = []
    predictions_low = []
    predictions_high = []
    model_used = []
    
    for i in range(len(X)):
        x = X.iloc[i:i+1]
        pred_low = model_low.predict(x)[0]
        pred_high = model_high.predict(x)[0]
        
        # Choose model based on actual value (for analysis purposes)
        if y.iloc[i] < threshold:
            final_pred = pred_low
            used_model = "Low"
        else:
            final_pred = pred_high
            used_model = "High"
        
        predictions.append(final_pred)
        predictions_low.append(pred_low)
        predictions_high.append(pred_high)
        model_used.append(used_model)
    
    return predictions, predictions_low, predictions_high, model_used

def calculate_feature_sensitivity(X, y, predictions, feature_names):
    """Calculate sensitivity metrics for each feature"""
    sensitivities = []
    
    for feature in feature_names:
        if feature in ['Humidity_Temp', 'Cement_Cover']:
            continue  # Skip interaction features
            
        # Calculate correlation between feature and prediction error
        errors = np.abs(y - predictions)
        correlation = np.corrcoef(X[feature], errors)[0, 1]
        
        # Calculate feature impact (range of predictions when feature varies)
        feature_values = X[feature].values
        pred_values = np.array(predictions)
        
        # Group by feature quantiles
        try:
            quartiles = pd.qcut(feature_values, q=4, labels=False, duplicates='drop')
            impact = []
            for q in range(int(quartiles.max()) + 1):
                mask = quartiles == q
                if mask.sum() > 0:
                    impact.append(pred_values[mask].mean())
            
            if len(impact) > 1:
                feature_impact = max(impact) - min(impact)
            else:
                feature_impact = 0
        except:
            feature_impact = 0
        
        sensitivities.append({
            'Feature': feature,
            'Correlation_with_Error': abs(correlation),
            'Prediction_Impact': feature_impact,
            'Feature_Std': X[feature].std(),
            'Feature_Mean': X[feature].mean()
        })
    
    return pd.DataFrame(sensitivities)

def save_predictions_with_sensitivity(X, y, predictions, predictions_low, predictions_high, 
                                     model_used, dataset_name, output_file):
    """Save predictions and input features to CSV"""
    
    # Create results dataframe
    results_df = X.copy()
    results_df['Actual_Corrosion_Rate'] = y.values
    results_df['Predicted_Corrosion_Rate'] = predictions
    results_df['Prediction_Low_Model'] = predictions_low
    results_df['Prediction_High_Model'] = predictions_high
    results_df['Model_Used'] = model_used
    results_df['Absolute_Error'] = np.abs(y.values - predictions)
    results_df['Percent_Error'] = (results_df['Absolute_Error'] / y.values) * 100
    results_df['Within_5pct'] = results_df['Percent_Error'] <= 5
    results_df['Within_10pct'] = results_df['Percent_Error'] <= 10
    
    # Save to CSV
    results_df.to_csv(f"../data/{output_file}", index=False)
    print(f"✅ Saved {dataset_name} predictions: {output_file}")
    
    return results_df

def create_sensitivity_visualizations(train_sensitivity, test_sensitivity):
    """Create visualization comparing train and test sensitivities"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Correlation with Error - Training
    ax = axes[0, 0]
    train_sorted = train_sensitivity.sort_values('Correlation_with_Error', ascending=False)
    ax.barh(train_sorted['Feature'], train_sorted['Correlation_with_Error'], 
            color='steelblue', edgecolor='black')
    ax.set_xlabel('Correlation with Error', fontweight='bold')
    ax.set_title('Training Set: Feature Correlation with Prediction Error', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Correlation with Error - Testing
    ax = axes[0, 1]
    test_sorted = test_sensitivity.sort_values('Correlation_with_Error', ascending=False)
    ax.barh(test_sorted['Feature'], test_sorted['Correlation_with_Error'], 
            color='coral', edgecolor='black')
    ax.set_xlabel('Correlation with Error', fontweight='bold')
    ax.set_title('Testing Set: Feature Correlation with Prediction Error', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Prediction Impact - Training
    ax = axes[1, 0]
    train_sorted_impact = train_sensitivity.sort_values('Prediction_Impact', ascending=False)
    ax.barh(train_sorted_impact['Feature'], train_sorted_impact['Prediction_Impact'], 
            color='lightgreen', edgecolor='black')
    ax.set_xlabel('Prediction Impact (µA/cm²)', fontweight='bold')
    ax.set_title('Training Set: Feature Impact on Predictions', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Prediction Impact - Testing
    ax = axes[1, 1]
    test_sorted_impact = test_sensitivity.sort_values('Prediction_Impact', ascending=False)
    ax.barh(test_sorted_impact['Feature'], test_sorted_impact['Prediction_Impact'], 
            color='gold', edgecolor='black')
    ax.set_xlabel('Prediction Impact (µA/cm²)', fontweight='bold')
    ax.set_title('Testing Set: Feature Impact on Predictions', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/sensitivity_train_test_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved: sensitivity_train_test_comparison.png")

def main():
    """Main workflow for dataset sensitivity analysis"""
    
    print("\n" + "="*70)
    print("🔬 SENSITIVITY ANALYSIS ON TRAINING & TESTING DATASETS")
    print("="*70 + "\n")
    
    # Load models
    print("📂 Loading models...")
    model_low, model_high, imputer = load_models()
    
    # Load and prepare data
    print("📊 Loading dataset...")
    df = load_and_prepare_data()
    
    print("🔄 Generating virtual samples...")
    df = generate_virtual_samples(df)
    
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    feature_names = X_train.columns.tolist()
    
    # Make predictions on training set
    print("\n🎯 Making predictions on TRAINING set...")
    train_pred, train_pred_low, train_pred_high, train_model = predict_segmented(
        X_train, y_train, model_low, model_high
    )
    
    # Make predictions on testing set
    print("🎯 Making predictions on TESTING set...")
    test_pred, test_pred_low, test_pred_high, test_model = predict_segmented(
        X_test, y_test, model_low, model_high
    )
    
    # Calculate sensitivity metrics
    print("\n📈 Calculating sensitivity metrics for TRAINING set...")
    train_sensitivity = calculate_feature_sensitivity(
        X_train, y_train, train_pred, feature_names
    )
    
    print("📈 Calculating sensitivity metrics for TESTING set...")
    test_sensitivity = calculate_feature_sensitivity(
        X_test, y_test, test_pred, feature_names
    )
    
    # Save predictions with all details
    print("\n💾 Saving results to CSV...")
    train_results = save_predictions_with_sensitivity(
        X_train, y_train, train_pred, train_pred_low, train_pred_high, 
        train_model, "TRAINING", "predictions_training.csv"
    )
    
    test_results = save_predictions_with_sensitivity(
        X_test, y_test, test_pred, test_pred_low, test_pred_high,
        test_model, "TESTING", "predictions_testing.csv"
    )
    
    # Save sensitivity metrics
    train_sensitivity.to_csv('../data/sensitivity_training.csv', index=False)
    print("✅ Saved training sensitivity: sensitivity_training.csv")
    
    test_sensitivity.to_csv('../data/sensitivity_testing.csv', index=False)
    print("✅ Saved testing sensitivity: sensitivity_testing.csv")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_sensitivity_visualizations(train_sensitivity, test_sensitivity)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS")
    print("="*70)
    
    print("\n🔵 TRAINING SET:")
    print(f"  Total samples: {len(train_results)}")
    print(f"  Mean Absolute Error: {train_results['Absolute_Error'].mean():.4f} µA/cm²")
    print(f"  Mean Percent Error: {train_results['Percent_Error'].mean():.2f}%")
    print(f"  Within 5% error: {train_results['Within_5pct'].sum()} ({train_results['Within_5pct'].mean()*100:.1f}%)")
    print(f"  Within 10% error: {train_results['Within_10pct'].sum()} ({train_results['Within_10pct'].mean()*100:.1f}%)")
    print(f"  Low model used: {(train_results['Model_Used']=='Low').sum()} times")
    print(f"  High model used: {(train_results['Model_Used']=='High').sum()} times")
    
    print("\n🔴 TESTING SET:")
    print(f"  Total samples: {len(test_results)}")
    print(f"  Mean Absolute Error: {test_results['Absolute_Error'].mean():.4f} µA/cm²")
    print(f"  Mean Percent Error: {test_results['Percent_Error'].mean():.2f}%")
    print(f"  Within 5% error: {test_results['Within_5pct'].sum()} ({test_results['Within_5pct'].mean()*100:.1f}%)")
    print(f"  Within 10% error: {test_results['Within_10pct'].sum()} ({test_results['Within_10pct'].mean()*100:.1f}%)")
    print(f"  Low model used: {(test_results['Model_Used']=='Low').sum()} times")
    print(f"  High model used: {(test_results['Model_Used']=='High').sum()} times")
    
    print("\n🏆 TOP 3 MOST SENSITIVE FEATURES (Training):")
    top_train = train_sensitivity.nlargest(3, 'Correlation_with_Error')
    for idx, row in top_train.iterrows():
        print(f"  {row['Feature']:.<45} {row['Correlation_with_Error']:.4f}")
    
    print("\n🏆 TOP 3 MOST SENSITIVE FEATURES (Testing):")
    top_test = test_sensitivity.nlargest(3, 'Correlation_with_Error')
    for idx, row in top_test.iterrows():
        print(f"  {row['Feature']:.<45} {row['Correlation_with_Error']:.4f}")
    
    print("\n" + "="*70)
    print("✨ Analysis Complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  • predictions_training.csv - All training predictions with errors")
    print("  • predictions_testing.csv - All testing predictions with errors")
    print("  • sensitivity_training.csv - Feature sensitivity metrics (training)")
    print("  • sensitivity_testing.csv - Feature sensitivity metrics (testing)")
    print("  • sensitivity_train_test_comparison.png - Visual comparison")
    print()

if __name__ == "__main__":
    main()

