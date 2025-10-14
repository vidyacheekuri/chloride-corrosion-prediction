"""
Chloride Corrosion XGBoost Model - Training Script
Trains a segmented XGBoost model for corrosion rate prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib


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
    
    return X_train, X_test, y_train, y_test, imp


def train_segmented_models(X_train, y_train, threshold=0.15):
    """Train separate models for low and high corrosion rates"""
    low_mask = y_train < threshold
    high_mask = y_train >= threshold
    
    model_low = XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )
    model_high = XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )
    
    print("Training low-rate model...")
    model_low.fit(X_train[low_mask], y_train[low_mask])
    
    print("Training high-rate model...")
    model_high.fit(X_train[high_mask], y_train[high_mask])
    
    return model_low, model_high


def evaluate_model(X_test, y_test, model_low, model_high, threshold=0.15):
    """Evaluate segmented model performance"""
    y_pred = []
    for i in range(len(X_test)):
        x = X_test.iloc[i:i+1]
        if y_test.iloc[i] < threshold:
            pred = model_low.predict(x)[0]
        else:
            pred = model_high.predict(x)[0]
        y_pred.append(pred)
    y_pred = np.array(y_pred)
    
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    error_pct = np.abs((y_test - y_pred) / y_test) * 100
    within_10 = (error_pct < 10).mean() * 100
    within_5 = (error_pct < 5).mean() * 100
    
    print("\n" + "="*50)
    print("🔀 Segmented XGBoost Model Performance")
    print("="*50)
    print(f"✅ R² Score      : {r2:.3f}")
    print(f"📉 RMSE          : {rmse:.4f}")
    print(f"📉 MAE           : {mae:.4f}")
    print(f"🎯 ≤10% Error    : {within_10:.2f}%")
    print(f"🎯 ≤5% Error     : {within_5:.2f}%")
    print("="*50 + "\n")
    
    return y_pred


def plot_predictions(y_test, y_pred, save_path="../visualizations/predictions_plot.png"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6, edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Corrosion Rate (μA/cm²)")
    plt.ylabel("Predicted Corrosion Rate (μA/cm²)")
    plt.title("Actual vs. Predicted Corrosion Rate (XGBoost)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")


def error_analysis(y_test, y_pred):
    """Perform detailed error analysis"""
    error_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    error_df["Absolute_Error"] = np.abs(error_df["Actual"] - error_df["Predicted"])
    error_df["Percent_Error"] = 100 * error_df["Absolute_Error"] / error_df["Actual"]
    
    # Distribution of Percent Error
    plt.figure(figsize=(7, 4))
    sns.histplot(error_df["Percent_Error"], bins=20, kde=True, color="purple")
    plt.title("Distribution of Percentage Error")
    plt.xlabel("Percent Error (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../visualizations/error_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error vs. Actual Corrosion Rate
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x=error_df["Actual"], y=error_df["Percent_Error"], alpha=0.6)
    plt.axhline(y=10, color='red', linestyle='--', label="10% Error Threshold")
    plt.xlabel("Actual Corrosion Rate (µA/cm²)")
    plt.ylabel("Percent Error (%)")
    plt.title("Percent Error vs. Actual Corrosion Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../visualizations/error_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error Summary
    within_5 = np.mean(error_df["Percent_Error"] <= 5) * 100
    within_10 = np.mean(error_df["Percent_Error"] <= 10) * 100
    within_30 = np.mean(error_df["Percent_Error"] <= 30) * 100
    mean_error = error_df["Percent_Error"].mean()
    median_error = error_df["Percent_Error"].median()
    
    print("\n" + "="*50)
    print("📊 Error Analysis Summary")
    print("="*50)
    print(f"Mean Percent Error    : {mean_error:.2f}%")
    print(f"Median Percent Error  : {median_error:.2f}%")
    print(f"Samples ≤ 5% Error     : {within_5:.2f}%")
    print(f"Samples ≤ 10% Error    : {within_10:.2f}%")
    print(f"Samples ≤ 30% Error    : {within_30:.2f}%")
    print("="*50 + "\n")


def save_models(model_low, model_high, imputer):
    """Save trained models and imputer"""
    joblib.dump(model_low, "../models/model_low.pkl")
    joblib.dump(model_high, "../models/model_high.pkl")
    joblib.dump(imputer, "../models/imputer.pkl")
    print("✅ Models saved to models/ folder")


def main():
    """Main training pipeline"""
    print("🚀 Starting Chloride Corrosion Model Training\n")
    
    # Load and prepare data
    print("📂 Loading dataset...")
    df = load_and_prepare_data()
    
    # Generate virtual samples
    print("🔄 Generating virtual samples...")
    df = generate_virtual_samples(df)
    
    # Prepare train-test split
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test, imp = prepare_train_test_split(df)
    
    # Train models
    print("🏋️  Training segmented models...")
    model_low, model_high = train_segmented_models(X_train, y_train)
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = evaluate_model(X_test, y_test, model_low, model_high)
    
    # Visualizations
    print("📈 Generating visualizations...")
    plot_predictions(y_test, y_pred)
    error_analysis(y_test, y_pred)
    
    # Save models
    print("💾 Saving models...")
    save_models(model_low, model_high, imp)
    
    print("\n✨ Training complete!")


if __name__ == "__main__":
    main()

