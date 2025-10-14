"""
Sensitivity Analysis for Chloride Corrosion Model
Analyzes how each input parameter affects the predicted corrosion rate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

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

def predict_corrosion(input_data, model_low, model_high, imputer, threshold=0.15):
    """Make prediction using the segmented models"""
    # Create interaction features
    input_data["Cement_Cover"] = input_data["Cover_Thickness_mm"] * input_data["Water_Cement_Ratio"]
    input_data["Humidity_Temp"] = input_data["Relative_Humidity_pct"] * input_data["Temperature_K"]
    
    # Define feature columns
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
    
    # Get predictions
    pred_low = model_low.predict(X_input)[0]
    pred_high = model_high.predict(X_input)[0]
    
    # Choose final prediction
    final_pred = pred_low if pred_low <= threshold else pred_high
    
    return final_pred

def get_baseline_values():
    """Define baseline values for sensitivity analysis"""
    return {
        "Cover_Thickness_mm": 50.0,
        "Reinforcement_Diameter_mm": 16.0,
        "Water_Cement_Ratio": 0.45,
        "Temperature_K": 298.0,
        "Relative_Humidity_pct": 75.0,
        "Chloride_Ion_Content_kgm3": 2.5,
        "Time_Years": 10.0
    }

def get_parameter_ranges():
    """Define ranges for each parameter"""
    return {
        "Cover_Thickness_mm": (20, 100, "Cover Thickness (mm)"),
        "Reinforcement_Diameter_mm": (8, 32, "Reinforcement Diameter (mm)"),
        "Water_Cement_Ratio": (0.3, 0.7, "Water-Cement Ratio"),
        "Temperature_K": (273, 323, "Temperature (K)"),
        "Relative_Humidity_pct": (40, 100, "Relative Humidity (%)"),
        "Chloride_Ion_Content_kgm3": (0.5, 6.0, "Chloride Content (kg/m³)"),
        "Time_Years": (1, 50, "Exposure Time (years)")
    }

def perform_sensitivity_analysis(model_low, model_high, imputer):
    """Perform sensitivity analysis for all parameters"""
    baseline = get_baseline_values()
    ranges = get_parameter_ranges()
    
    results = {}
    n_points = 50  # Number of points to test for each parameter
    
    print("🔬 Performing Sensitivity Analysis...")
    print("=" * 60)
    
    for param, (min_val, max_val, display_name) in ranges.items():
        print(f"Analyzing: {display_name}")
        
        # Generate test values
        test_values = np.linspace(min_val, max_val, n_points)
        predictions = []
        
        # Test each value
        for val in test_values:
            # Create input with baseline values
            input_dict = baseline.copy()
            input_dict[param] = val
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Predict
            pred = predict_corrosion(input_df, model_low, model_high, imputer)
            predictions.append(pred)
        
        # Calculate sensitivity metrics
        pred_range = max(predictions) - min(predictions)
        relative_change = (pred_range / predictions[0]) * 100 if predictions[0] != 0 else 0
        
        results[param] = {
            'values': test_values,
            'predictions': predictions,
            'display_name': display_name,
            'range': pred_range,
            'relative_change': relative_change,
            'baseline_value': baseline[param]
        }
        
        print(f"  → Prediction range: {pred_range:.4f} µA/cm²")
        print(f"  → Relative change: {relative_change:.2f}%")
    
    return results

def plot_sensitivity_analysis(results):
    """Create comprehensive sensitivity analysis plots"""
    n_params = len(results)
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Individual parameter plots
    for idx, (param, data) in enumerate(results.items()):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Plot
        ax.plot(data['values'], data['predictions'], 'b-', linewidth=2)
        ax.axvline(data['baseline_value'], color='r', linestyle='--', 
                   alpha=0.7, label='Baseline')
        ax.axhline(data['predictions'][0], color='g', linestyle=':', 
                   alpha=0.5, label='Baseline Prediction')
        
        # Styling
        ax.set_xlabel(data['display_name'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Corrosion Rate (µA/cm²)', fontsize=10)
        ax.set_title(f"Sensitivity: {data['display_name']}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add annotation for range
        ax.text(0.05, 0.95, f"Range: {data['range']:.4f} µA/cm²\nChange: {data['relative_change']:.1f}%",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
    
    plt.suptitle('Sensitivity Analysis: Effect of Each Parameter on Corrosion Rate', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('../visualizations/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Plot saved: sensitivity_analysis.png")
    
    return fig

def plot_sensitivity_ranking(results):
    """Create a ranking plot of parameter sensitivities"""
    # Extract data for ranking
    params = []
    relative_changes = []
    
    for param, data in results.items():
        params.append(data['display_name'])
        relative_changes.append(data['relative_change'])
    
    # Sort by sensitivity
    sorted_indices = np.argsort(relative_changes)[::-1]
    params_sorted = [params[i] for i in sorted_indices]
    changes_sorted = [relative_changes[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(params_sorted)))
    
    bars = ax.barh(params_sorted, changes_sorted, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, changes_sorted)):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Relative Change in Prediction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Sensitivity Ranking\n(Higher = More Sensitive)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/sensitivity_ranking.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved: sensitivity_ranking.png")
    
    return fig

def generate_sensitivity_report(results):
    """Generate a text report of sensitivity analysis"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SENSITIVITY ANALYSIS REPORT")
    report_lines.append("Chloride Corrosion Prediction Model")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Baseline prediction
    baseline = get_baseline_values()
    baseline_df = pd.DataFrame([baseline])
    model_low, model_high, imputer = load_models()
    baseline_pred = predict_corrosion(baseline_df, model_low, model_high, imputer)
    
    report_lines.append("BASELINE CONDITIONS:")
    report_lines.append("-" * 70)
    for param, value in baseline.items():
        param_name = results[param]['display_name']
        report_lines.append(f"  {param_name:.<40} {value:>10.2f}")
    report_lines.append(f"\n  {'Baseline Prediction':.<40} {baseline_pred:>10.4f} µA/cm²")
    report_lines.append("")
    
    # Sensitivity ranking
    report_lines.append("\nSENSITIVITY RANKING (Most to Least Influential):")
    report_lines.append("-" * 70)
    
    # Sort by relative change
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['relative_change'], 
                          reverse=True)
    
    for rank, (param, data) in enumerate(sorted_results, 1):
        report_lines.append(f"\n{rank}. {data['display_name']}")
        report_lines.append(f"   Prediction Range: {data['range']:.4f} µA/cm²")
        report_lines.append(f"   Relative Change:  {data['relative_change']:.2f}%")
        
        # Determine impact level
        if data['relative_change'] > 50:
            impact = "Very High"
        elif data['relative_change'] > 25:
            impact = "High"
        elif data['relative_change'] > 10:
            impact = "Moderate"
        else:
            impact = "Low"
        report_lines.append(f"   Impact Level:     {impact}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("\nKEY INSIGHTS:")
    report_lines.append("-" * 70)
    
    # Top 3 most sensitive
    top_3 = sorted_results[:3]
    report_lines.append("\nMost Influential Parameters:")
    for rank, (param, data) in enumerate(top_3, 1):
        report_lines.append(f"  {rank}. {data['display_name']} ({data['relative_change']:.1f}% change)")
    
    # Least sensitive
    least_sensitive = sorted_results[-1]
    report_lines.append(f"\nLeast Influential Parameter:")
    report_lines.append(f"  → {least_sensitive[1]['display_name']} ({least_sensitive[1]['relative_change']:.1f}% change)")
    
    report_lines.append("\n" + "=" * 70)
    
    # Save report
    report_text = "\n".join(report_lines)
    with open('../data/sensitivity_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n✅ Report saved: sensitivity_report.txt")
    print("\n" + report_text)
    
    return report_text

def main():
    """Main sensitivity analysis workflow"""
    print("\n" + "=" * 60)
    print("🔬 SENSITIVITY ANALYSIS")
    print("Chloride Corrosion Prediction Model")
    print("=" * 60 + "\n")
    
    # Load models
    print("📂 Loading models...")
    model_low, model_high, imputer = load_models()
    
    # Perform analysis
    results = perform_sensitivity_analysis(model_low, model_high, imputer)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_sensitivity_analysis(results)
    plot_sensitivity_ranking(results)
    
    # Generate report
    print("\n📝 Generating report...")
    generate_sensitivity_report(results)
    
    print("\n" + "=" * 60)
    print("✨ Sensitivity Analysis Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • sensitivity_analysis.png - Individual parameter effects")
    print("  • sensitivity_ranking.png - Parameter importance ranking")
    print("  • sensitivity_report.txt - Detailed text report")
    print()

if __name__ == "__main__":
    main()

