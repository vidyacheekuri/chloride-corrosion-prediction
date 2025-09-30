"""
Sensitivity Analysis Module for Chloride Corrosion Prediction

This module performs comprehensive input variable sensitivity analysis to understand
how changes in each input variable affect corrosion rate predictions.

Author: Research Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from typing import List, Dict, Any, Tuple
import joblib

warnings.filterwarnings('ignore')


class SensitivityAnalyzer:
    """
    A class for performing input variable sensitivity analysis on chloride corrosion models.
    
    This class analyzes how perturbations in input variables affect model predictions
    and provides comprehensive visualizations and insights.
    """
    
    def __init__(self, perturbation_levels: List[float] = None):
        """
        Initialize the SensitivityAnalyzer.
        
        Args:
            perturbation_levels (List[float]): List of perturbation percentages (default: [-10%, -5%, +5%, +10%])
        """
        self.perturbation_levels = perturbation_levels or [-0.10, -0.05, +0.05, +0.10]
        self.feature_columns = [
            'Cover_Thickness_mm', 'Reinforcement_Diameter_mm', 'Water_Cement_Ratio',
            'Temperature_K', 'Relative_Humidity_pct', 'Chloride_Ion_Content_kgm3', 
            'Time_Years'
        ]
        self.scaler = None
        self.baseline_model = None
        self.sensitivity_results = None
        self.detailed_results = None
        
    def load_data_and_model(self, train_path: str, test_path: str, 
                           model_path: str = None, scaler_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data and trained model.
        
        Args:
            train_path (str): Path to training data (for retraining if model not provided)
            test_path (str): Path to test data
            model_path (str): Path to saved model (optional)
            scaler_path (str): Path to saved scaler (optional)
            
        Returns:
            Tuple of (X_test_scaled, y_test)
        """
        print("=== INPUT VARIABLE SENSITIVITY ANALYSIS ===")
        print("Analyzing how changes in each input variable affect corrosion rate predictions")

        # Load test data
        test_data = pd.read_csv(test_path)
        X_test = test_data[self.feature_columns]
        y_test = test_data['Corrosion_Rate_uAcm2']
        
        print(f"Loaded test data: {len(test_data)} samples")
        
        if model_path and scaler_path and os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load pre-trained model and scaler
            print("Loading pre-trained model and scaler...")
            self.baseline_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Pre-trained model loaded successfully!")
        else:
            # Train new model
            print("Training new baseline model...")
            self._train_baseline_model(train_path)
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_test_scaled, y_test
    
    def _train_baseline_model(self, train_path: str) -> None:
        """
        Train baseline model for sensitivity analysis.
        
        Args:
            train_path (str): Path to training data
        """
        print("\n--- Training Baseline Model ---")
        
        # Load training data
        train_augmented = pd.read_csv(train_path)
        X_train_aug = train_augmented[self.feature_columns]
        y_train_log_aug = train_augmented['log_Corrosion_Rate_uAcm2']
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_aug)
        
        # Create and train ensemble model
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25), activation='relu', alpha=0.001,
            learning_rate_init=0.005, max_iter=1500, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25
        )

        xgb_model1 = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            reg_alpha=0.1, reg_lambda=0.1, min_child_weight=3
        )

        xgb_model2 = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=123,
            reg_alpha=0.05, reg_lambda=0.05, min_child_weight=1
        )

        self.baseline_model = VotingRegressor(
            estimators=[('mlp', mlp_model), ('xgb1', xgb_model1), ('xgb2', xgb_model2)],
            weights=[0.3, 0.4, 0.3]
        )

        self.baseline_model.fit(X_train_scaled, y_train_log_aug)
        print("Baseline model trained successfully!")
    
    def get_baseline_predictions(self, X_test_scaled: np.ndarray) -> np.ndarray:
        """
        Get baseline predictions for all test samples.
        
        Args:
            X_test_scaled (np.ndarray): Scaled test features
            
        Returns:
            np.ndarray: Baseline predictions
        """
        baseline_pred_log = self.baseline_model.predict(X_test_scaled)
        baseline_predictions = np.expm1(baseline_pred_log)
        
        print(f"Baseline predictions computed for {len(baseline_predictions)} test samples")
        return baseline_predictions
    
    def perform_sensitivity_analysis(self, X_test: pd.DataFrame, X_test_scaled: np.ndarray, 
                                   y_test: pd.Series, baseline_predictions: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform comprehensive sensitivity analysis.
        
        Args:
            X_test (pd.DataFrame): Original test features
            X_test_scaled (np.ndarray): Scaled test features
            y_test (pd.Series): Test targets
            baseline_predictions (np.ndarray): Baseline predictions
            
        Returns:
            Tuple of (sensitivity_df, detailed_df)
        """
        print(f"\n--- SENSITIVITY ANALYSIS PROCEDURE ---")
        print(f"Perturbation levels: {[f'{p*100:+.0f}%' for p in self.perturbation_levels]}")
        print(f"Testing each input variable separately (one at a time)")

        # Initialize results storage
        sensitivity_results = []
        detailed_results = []

        # Perform sensitivity analysis for each feature
        for feature_idx, feature_name in enumerate(self.feature_columns):
            print(f"\nAnalyzing sensitivity of: {feature_name}")

            feature_sensitivities = []

            # Test each perturbation level
            for perturbation in self.perturbation_levels:
                # Create perturbed version of test data
                X_test_perturbed = X_test.copy()

                # Apply perturbation to the current feature only
                original_values = X_test_perturbed.iloc[:, feature_idx].values
                perturbed_values = original_values * (1 + perturbation)
                X_test_perturbed.iloc[:, feature_idx] = perturbed_values

                # Scale the perturbed data
                X_test_perturbed_scaled = self.scaler.transform(X_test_perturbed)

                # Make predictions with perturbed data
                perturbed_pred_log = self.baseline_model.predict(X_test_perturbed_scaled)
                perturbed_predictions = np.expm1(perturbed_pred_log)

                # Calculate impact for each test sample
                for sample_idx in range(len(baseline_predictions)):
                    baseline_pred = baseline_predictions[sample_idx]
                    perturbed_pred = perturbed_predictions[sample_idx]
                    actual_value = y_test.iloc[sample_idx]

                    # Calculate percentage impact on prediction
                    if baseline_pred != 0:
                        impact_on_prediction = ((perturbed_pred - baseline_pred) / baseline_pred) * 100
                    else:
                        impact_on_prediction = 0

                    # Calculate error change relative to actual
                    baseline_error = abs((baseline_pred - actual_value) / actual_value) * 100
                    perturbed_error = abs((perturbed_pred - actual_value) / actual_value) * 100
                    error_change = perturbed_error - baseline_error

                    # Store detailed results
                    detailed_results.append({
                        'Feature': feature_name,
                        'Perturbation_%': perturbation * 100,
                        'Sample_Index': sample_idx,
                        'Original_Feature_Value': original_values[sample_idx],
                        'Perturbed_Feature_Value': perturbed_values[sample_idx],
                        'Baseline_Prediction': baseline_pred,
                        'Perturbed_Prediction': perturbed_pred,
                        'Actual_Value': actual_value,
                        'Impact_on_Prediction_%': impact_on_prediction,
                        'Baseline_Error_%': baseline_error,
                        'Perturbed_Error_%': perturbed_error,
                        'Error_Change_%': error_change
                    })

                # Calculate average impact across all samples
                avg_impact = np.mean([
                    ((perturbed_predictions[i] - baseline_predictions[i]) / baseline_predictions[i]) * 100
                    for i in range(len(baseline_predictions)) if baseline_predictions[i] != 0
                ])

                feature_sensitivities.append(abs(avg_impact))

                print(f"   {perturbation*100:+3.0f}%: Avg impact = {avg_impact:+6.2f}%")

            # Store overall sensitivity for this feature
            max_sensitivity = max(feature_sensitivities)
            avg_sensitivity = np.mean(feature_sensitivities)

            sensitivity_results.append({
                'Feature': feature_name,
                'Max_Sensitivity_%': max_sensitivity,
                'Avg_Sensitivity_%': avg_sensitivity,
                'Sensitivity_Rank': 0  # Will be filled later
            })

        # Convert to DataFrames
        sensitivity_df = pd.DataFrame(sensitivity_results)
        detailed_df = pd.DataFrame(detailed_results)

        # Rank features by sensitivity
        sensitivity_df = sensitivity_df.sort_values('Max_Sensitivity_%', ascending=False)
        sensitivity_df['Sensitivity_Rank'] = range(1, len(sensitivity_df) + 1)

        # Store results
        self.sensitivity_results = sensitivity_df
        self.detailed_results = detailed_df

        return sensitivity_df, detailed_df
    
    def print_sensitivity_ranking(self) -> None:
        """Print sensitivity ranking results."""
        if self.sensitivity_results is None:
            print("No sensitivity results available. Run analysis first.")
            return
            
        print(f"\n--- SENSITIVITY RANKING ---")
        print(f"Features ranked by maximum sensitivity to perturbations:")
        for idx, row in self.sensitivity_results.iterrows():
            print(f"{row['Sensitivity_Rank']:2d}. {row['Feature']:<30} Max: {row['Max_Sensitivity_%']:6.2f}% | Avg: {row['Avg_Sensitivity_%']:6.2f}%")
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create detailed sensitivity summary table.
        
        Returns:
            pd.DataFrame: Summary table
        """
        if self.detailed_results is None:
            print("No detailed results available. Run analysis first.")
            return pd.DataFrame()
            
        print(f"\n--- DETAILED SENSITIVITY TABLE ---")
        summary_table = []

        for feature_name in self.feature_columns:
            feature_data = self.detailed_results[self.detailed_results['Feature'] == feature_name]

            for perturbation in self.perturbation_levels:
                pert_data = feature_data[feature_data['Perturbation_%'] == perturbation * 100]

                avg_baseline = pert_data['Baseline_Prediction'].mean()
                avg_perturbed = pert_data['Perturbed_Prediction'].mean()
                avg_impact = pert_data['Impact_on_Prediction_%'].mean()

                summary_table.append({
                    'Input_Variable': feature_name,
                    'Variation_%': f"{perturbation*100:+.0f}%",
                    'Avg_Predicted_Corrosion_Rate': f"{avg_perturbed:.4f}",
                    'Impact_on_Prediction_%': f"{avg_impact:+.2f}%"
                })

        summary_df = pd.DataFrame(summary_table)
        
        # Display summary table
        print("\nSUMMARY TABLE:")
        print("=" * 100)
        print(f"{'Input Variable':<25} {'Variation (%)':<12} {'Avg Predicted Rate':<20} {'Impact on Prediction (%)':<25}")
        print("=" * 100)
        for _, row in summary_df.iterrows():
            print(f"{row['Input_Variable']:<25} {row['Variation_%']:<12} {row['Avg_Predicted_Corrosion_Rate']:<20} {row['Impact_on_Prediction_%']:<25}")
        
        return summary_df
    
    def create_visualizations(self, output_dir: str = "results") -> None:
        """
        Create comprehensive sensitivity analysis visualizations.
        
        Args:
            output_dir (str): Directory to save plots
        """
        if self.sensitivity_results is None or self.detailed_results is None:
            print("No sensitivity results available. Run analysis first.")
            return
            
        print(f"\n--- CREATING VISUALIZATIONS ---")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Tornado diagram showing sensitivity ranking
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(self.sensitivity_results))
        plt.barh(y_pos, self.sensitivity_results['Max_Sensitivity_%'], 
                color='skyblue', edgecolor='navy')
        plt.yticks(y_pos, self.sensitivity_results['Feature'])
        plt.xlabel('Maximum Sensitivity (%)')
        plt.title('Feature Sensitivity Ranking\n(How much each input affects corrosion rate predictions)')
        plt.grid(axis='x', alpha=0.3)

        # Add values on bars
        for i, (idx, row) in enumerate(self.sensitivity_results.iterrows()):
            plt.text(row['Max_Sensitivity_%'] + 0.1, i, f"{row['Max_Sensitivity_%']:.1f}%",
                     va='center', fontweight='bold')

        plt.tight_layout()
        ranking_path = os.path.join(output_dir, 'sensitivity_ranking.png')
        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Detailed sensitivity heatmap
        print(f"Creating detailed sensitivity heatmap...")

        # Pivot data for heatmap
        pivot_data = self.detailed_results.groupby(['Feature', 'Perturbation_%'])['Impact_on_Prediction_%'].mean().unstack()
        pivot_data = pivot_data.reindex(self.sensitivity_results['Feature'])  # Order by sensitivity

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                    cbar_kws={'label': 'Impact on Prediction (%)'})
        plt.title('Input Variable Sensitivity Heatmap\n(Average impact across all test samples)')
        plt.xlabel('Perturbation Level (%)')
        plt.ylabel('Input Variables (ordered by sensitivity)')
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'sensitivity_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Individual feature sensitivity plots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, feature_name in enumerate(self.feature_columns):
            if i < len(axes):
                feature_data = self.detailed_results[self.detailed_results['Feature'] == feature_name]
                perturbation_impacts = feature_data.groupby('Perturbation_%')['Impact_on_Prediction_%'].mean()

                axes[i].plot(perturbation_impacts.index, perturbation_impacts.values,
                            'o-', linewidth=2, markersize=6, color='red')
                axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3)
                axes[i].set_xlabel('Perturbation (%)')
                axes[i].set_ylabel('Impact on Prediction (%)')
                axes[i].set_title(f'{feature_name}')
                axes[i].grid(True, alpha=0.3)

        plt.suptitle('Individual Feature Sensitivity Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        individual_path = os.path.join(output_dir, 'individual_sensitivities.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to:")
        print(f"   - {ranking_path}")
        print(f"   - {heatmap_path}")
        print(f"   - {individual_path}")
    
    def print_insights(self) -> None:
        """Print sensitivity analysis insights and recommendations."""
        if self.sensitivity_results is None:
            print("No sensitivity results available. Run analysis first.")
            return
            
        print(f"\n--- SENSITIVITY ANALYSIS INSIGHTS ---")

        most_sensitive = self.sensitivity_results.iloc[0]
        least_sensitive = self.sensitivity_results.iloc[-1]

        print(f"MOST SENSITIVE FEATURE: {most_sensitive['Feature']}")
        print(f"   Maximum impact: {most_sensitive['Max_Sensitivity_%']:.2f}%")
        print(f"   This means small changes in {most_sensitive['Feature']} significantly affect predictions")

        print(f"\nLEAST SENSITIVE FEATURE: {least_sensitive['Feature']}")
        print(f"   Maximum impact: {least_sensitive['Max_Sensitivity_%']:.2f}%")
        print(f"   This feature has minimal impact on prediction changes")

        print(f"\nSENSITIVITY CATEGORIES:")
        high_sensitivity = self.sensitivity_results[self.sensitivity_results['Max_Sensitivity_%'] >= 5]
        medium_sensitivity = self.sensitivity_results[(self.sensitivity_results['Max_Sensitivity_%'] >= 2) & 
                                                    (self.sensitivity_results['Max_Sensitivity_%'] < 5)]
        low_sensitivity = self.sensitivity_results[self.sensitivity_results['Max_Sensitivity_%'] < 2]

        print(f"   High Sensitivity (≥5%): {len(high_sensitivity)} features")
        for _, row in high_sensitivity.iterrows():
            print(f"     - {row['Feature']}: {row['Max_Sensitivity_%']:.1f}%")

        print(f"   Medium Sensitivity (2-5%): {len(medium_sensitivity)} features")
        for _, row in medium_sensitivity.iterrows():
            print(f"     - {row['Feature']}: {row['Max_Sensitivity_%']:.1f}%")

        print(f"   Low Sensitivity (<2%): {len(low_sensitivity)} features")
        for _, row in low_sensitivity.iterrows():
            print(f"     - {row['Feature']}: {row['Max_Sensitivity_%']:.1f}%")

        # Practical implications
        print(f"\n--- PRACTICAL IMPLICATIONS ---")
        print(f"For Model Reliability:")
        print(f"   • Focus measurement accuracy on: {', '.join(high_sensitivity['Feature'].tolist())}")
        print(f"   • Less critical measurements: {', '.join(low_sensitivity['Feature'].tolist())}")

        print(f"\nFor Feature Engineering:")
        high_sens_features = high_sensitivity['Feature'].tolist()
        if len(high_sens_features) > 0:
            print(f"   • Consider creating interaction terms with: {', '.join(high_sens_features[:2])}")
            print(f"   • Apply extra preprocessing/validation to: {', '.join(high_sens_features)}")

        print(f"\nFor Data Collection:")
        print(f"   • Prioritize data quality for high-sensitivity features")
        print(f"   • Consider collecting more samples with varied {most_sensitive['Feature']} values")
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save sensitivity analysis results.
        
        Args:
            output_dir (str): Directory to save results
        """
        if self.sensitivity_results is None or self.detailed_results is None:
            print("No sensitivity results available. Run analysis first.")
            return
            
        print(f"\n--- Saving Results ---")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        sensitivity_path = os.path.join(output_dir, 'feature_sensitivity_ranking.csv')
        self.sensitivity_results.to_csv(sensitivity_path, index=False)
        
        detailed_path = os.path.join(output_dir, 'detailed_sensitivity_analysis.csv')
        self.detailed_results.to_csv(detailed_path, index=False)
        
        # Create and save summary table
        summary_df = self.create_summary_table()
        summary_path = os.path.join(output_dir, 'sensitivity_summary_table.csv')
        summary_df.to_csv(summary_path, index=False)

        print(f"Results saved to:")
        print(f"   - {sensitivity_path}")
        print(f"   - {detailed_path}")
        print(f"   - {summary_path}")
    
    def run_complete_analysis(self, train_path: str, test_path: str, 
                            model_path: str = None, scaler_path: str = None,
                            output_dir: str = "results") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete sensitivity analysis pipeline.
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data
            model_path (str): Path to saved model (optional)
            scaler_path (str): Path to saved scaler (optional)
            output_dir (str): Directory to save results
            
        Returns:
            Tuple of (sensitivity_df, detailed_df)
        """
        # Load data and model
        X_test_scaled, y_test = self.load_data_and_model(train_path, test_path, model_path, scaler_path)
        
        # Get baseline predictions
        baseline_predictions = self.get_baseline_predictions(X_test_scaled)
        
        # Load original test data for perturbation
        test_data = pd.read_csv(test_path)
        X_test = test_data[self.feature_columns]
        
        # Perform sensitivity analysis
        sensitivity_df, detailed_df = self.perform_sensitivity_analysis(X_test, X_test_scaled, y_test, baseline_predictions)
        
        # Print results
        self.print_sensitivity_ranking()
        self.create_summary_table()
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Print insights
        self.print_insights()
        
        # Save results
        self.save_results(output_dir)
        
        print(f"\nINPUT VARIABLE SENSITIVITY ANALYSIS COMPLETE!")
        print(f"Analyzed {len(self.feature_columns)} features × {len(self.perturbation_levels)} perturbations × {len(y_test)} samples")
        print(f"Total predictions made: {len(self.feature_columns) * len(self.perturbation_levels) * len(y_test) + len(y_test):,}")
        
        return sensitivity_df, detailed_df


def main():
    """Main function to run sensitivity analysis."""
    # Initialize analyzer
    analyzer = SensitivityAnalyzer()
    
    # Run complete analysis
    train_path = "../data/augmented_training_data_unscaled.csv"
    test_path = "../data/testing_data.csv"
    model_path = "../results/chloride_corrosion_model.pkl"
    scaler_path = "../results/feature_scaler.pkl"
    
    sensitivity_df, detailed_df = analyzer.run_complete_analysis(
        train_path, test_path, model_path, scaler_path
    )
    
    return sensitivity_df, detailed_df


if __name__ == "__main__":
    main()
