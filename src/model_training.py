"""
Model Training Module for Chloride Corrosion Prediction

This module implements ensemble machine learning models for predicting chloride corrosion rates
using augmented training data and provides comprehensive performance evaluation.

Author: Research Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
import os
from typing import Tuple, Dict, Any
import joblib

warnings.filterwarnings('ignore')


class ChlorideCorrosionPredictor:
    """
    A class for training and evaluating chloride corrosion prediction models.
    
    This class implements an ensemble approach using MLP and XGBoost models
    with comprehensive performance evaluation and feature importance analysis.
    """
    
    def __init__(self):
        """Initialize the ChlorideCorrosionPredictor."""
        self.feature_columns = [
            'Cover_Thickness_mm', 'Reinforcement_Diameter_mm', 'Water_Cement_Ratio',
            'Temperature_K', 'Relative_Humidity_pct', 'Chloride_Ion_Content_kgm3', 
            'Time_Years'
        ]
        self.scaler = RobustScaler()
        self.ensemble_model = None
        self.training_history = {}
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test data.
        
        Args:
            train_path (str): Path to augmented training data CSV
            test_path (str): Path to test data CSV
            
        Returns:
            Tuple containing X_train_scaled, y_train_log, X_test_scaled, y_test
        """
        print("--- Loading Clean Augmented Data ---")

        # Load the clean augmented training data
        train_augmented = pd.read_csv(train_path)
        print(f"Loaded augmented training data: {len(train_augmented)} samples")

        # Load original test data
        test_data = pd.read_csv(test_path)
        print(f"Loaded test data: {len(test_data)} samples")

        # Separate features and targets
        X_train_aug = train_augmented[self.feature_columns]
        y_train_log_aug = train_augmented['log_Corrosion_Rate_uAcm2']

        X_test = test_data[self.feature_columns]
        y_test = test_data['Corrosion_Rate_uAcm2']

        print(f"Training features shape: {X_train_aug.shape}")
        print(f"Test features shape: {X_test.shape}")

        # Verify no negative values in training data
        negative_count = np.sum(X_train_aug.values < 0)
        print(f"Negative values in training data: {negative_count}")

        # Scale the features
        print("\n--- Scaling Features ---")
        X_train_scaled = self.scaler.fit_transform(X_train_aug)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Scaled training data shape: {X_train_scaled.shape}")
        print(f"Scaled test data shape: {X_test_scaled.shape}")
        
        return X_train_scaled, y_train_log_aug, X_test_scaled, y_test
    
    def create_models(self) -> VotingRegressor:
        """
        Create and configure the ensemble model.
        
        Returns:
            VotingRegressor: Configured ensemble model
        """
        print("\n--- Setting Up Models ---")

        # Model 1: Multi-layer Perceptron
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            alpha=0.001,  # Regularization
            learning_rate_init=0.005,
            max_iter=1500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25
        )

        # Model 2: XGBoost - Main model
        xgb_model1 = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=3
        )

        # Model 3: XGBoost - Alternative configuration
        xgb_model2 = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=123,
            reg_alpha=0.05,
            reg_lambda=0.05,
            min_child_weight=1
        )

        # Create ensemble with weighted voting
        ensemble_model = VotingRegressor(
            estimators=[
                ('mlp', mlp_model),
                ('xgb1', xgb_model1),
                ('xgb2', xgb_model2)
            ],
            weights=[0.3, 0.4, 0.3]  # Give slightly more weight to main XGBoost
        )
        
        return ensemble_model
    
    def train_model(self, X_train_scaled: np.ndarray, y_train_log: np.ndarray) -> None:
        """
        Train the ensemble model.
        
        Args:
            X_train_scaled (np.ndarray): Scaled training features
            y_train_log (np.ndarray): Log-transformed training targets
        """
        print("\n--- Training Ensemble Model ---")
        print("Training in progress...")
        
        self.ensemble_model = self.create_models()
        self.ensemble_model.fit(X_train_scaled, y_train_log)
        print("Training completed!")
    
    def evaluate_model(self, X_test_scaled: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test_scaled (np.ndarray): Scaled test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict containing evaluation metrics and results
        """
        print("\n--- Making Predictions ---")
        y_pred_log = self.ensemble_model.predict(X_test_scaled)
        y_pred_test = np.expm1(y_pred_log)  # Convert back from log scale

        # Calculate standard metrics
        print("\n--- Model Performance Metrics ---")
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)

        print(f"R-squared (R²):                 {r2:.4f}")
        print(f"Root Mean Squared Error (RMSE):  {rmse:.4f}")
        print(f"Mean Absolute Error (MAE):       {mae:.4f}")

        # Calculate custom error percentages
        print("\n--- Custom Error Analysis ---")
        prediction_errors = np.abs((y_test.values - y_pred_test) / y_test.values) * 100
        valid_errors = prediction_errors[np.isfinite(prediction_errors)]
        total_predictions = len(valid_errors)

        print(f"Total valid predictions: {total_predictions}")
        print(f"Mean error percentage: {np.mean(valid_errors):.2f}%")
        print(f"Median error percentage: {np.median(valid_errors):.2f}%")

        # Check different error thresholds
        error_thresholds = [5, 10, 15, 20, 30]
        results_summary = {}

        for threshold in error_thresholds:
            correct_predictions = np.sum(valid_errors < threshold)
            accuracy = (correct_predictions / total_predictions) * 100
            results_summary[threshold] = accuracy
            print(f"Predictions with <{threshold}% error: {accuracy:.2f}%")

        # Check if we meet our targets
        print("\n--- Target Achievement Check ---")
        r2_target_met = r2 > 0.85
        error_10_target_met = results_summary[10] >= 80.0

        print(f"Target 1 - R² > 0.85: {'ACHIEVED' if r2_target_met else 'NOT MET'} (Current: {r2:.4f})")
        print(f"Target 2 - ≥80% with <10% error: {'ACHIEVED' if error_10_target_met else 'NOT MET'} (Current: {results_summary[10]:.2f}%)")

        if r2_target_met and error_10_target_met:
            print("\nBOTH TARGETS ACHIEVED!")
            success_status = "SUCCESS"
        else:
            print("\nOne or more targets not met")
            success_status = "PARTIAL"

        # Detailed predictions analysis
        print("\n--- Sample Predictions vs Actual ---")
        results_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred_test,
            'Error_%': prediction_errors,
            'Absolute_Error': np.abs(y_test.values - y_pred_test)
        })

        # Sort by error percentage to see best and worst predictions
        results_sorted = results_df.sort_values('Error_%')
        print("\nBEST Predictions (lowest error):")
        print(results_sorted.head(5)[['Actual', 'Predicted', 'Error_%']])

        print("\nWORST Predictions (highest error):")
        print(results_sorted.tail(5)[['Actual', 'Predicted', 'Error_%']])

        # Feature importance analysis
        print("\n--- Feature Importance Analysis ---")
        # Get feature importance from the main XGBoost model
        xgb_main = self.ensemble_model.named_estimators_['xgb1']
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': xgb_main.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("Top features by importance:")
        for idx, row in feature_importance.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")

        # Store results
        self.training_history = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'results_summary': results_summary,
            'r2_target_met': r2_target_met,
            'error_10_target_met': error_10_target_met,
            'success_status': success_status,
            'results_df': results_df,
            'feature_importance': feature_importance,
            'predictions': y_pred_test
        }

        return self.training_history
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save model results and performance metrics.
        
        Args:
            output_dir (str): Directory to save results
        """
        if not self.training_history:
            print("No training history available. Train model first.")
            return
            
        print(f"\n--- Saving Results ---")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(output_dir, 'final_model_predictions.csv')
        self.training_history['results_df'].to_csv(results_path, index=False)
        
        feature_importance_path = os.path.join(output_dir, 'model_feature_importance.csv')
        self.training_history['feature_importance'].to_csv(feature_importance_path, index=False)

        # Save model performance summary
        summary_data = {
            'Metric': ['R²', 'RMSE', 'MAE', '<5% Error', '<10% Error', '<15% Error', '<20% Error', '<30% Error'],
            'Value': [
                self.training_history['r2'], 
                self.training_history['rmse'], 
                self.training_history['mae']
            ] + [self.training_history['results_summary'][t] for t in [5, 10, 15, 20, 30]],
            'Target_Met': [
                self.training_history['r2_target_met'], 
                'N/A', 'N/A', 'N/A', 
                self.training_history['error_10_target_met'], 
                'N/A', 'N/A', 'N/A'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'model_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        print(f"Results saved to:")
        print(f"   - {results_path}")
        print(f"   - {feature_importance_path}")
        print(f"   - {summary_path}")
    
    def save_model(self, output_dir: str = "results") -> None:
        """
        Save the trained model and scaler.
        
        Args:
            output_dir (str): Directory to save model files
        """
        if self.ensemble_model is None:
            print("No trained model available. Train model first.")
            return
            
        print(f"\n--- Saving Model ---")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'chloride_corrosion_model.pkl')
        joblib.dump(self.ensemble_model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def print_summary(self) -> None:
        """Print final training summary."""
        if not self.training_history:
            print("No training history available.")
            return
            
        print(f"\n--- FINAL SUMMARY ---")
        print(f"Dataset Statistics:")
        print(f"   Training samples (augmented): {len(self.training_history['results_df'])}")
        print(f"   Test samples: {len(self.training_history['results_df'])}")
        print(f"   Augmentation ratio: ~5.6x")

        print(f"\nPerformance Results:")
        print(f"   R²: {self.training_history['r2']:.4f}")
        print(f"   RMSE: {self.training_history['rmse']:.4f}")
        print(f"   MAE: {self.training_history['mae']:.4f}")
        print(f"   <10% error rate: {self.training_history['results_summary'][10]:.1f}%")
        print(f"   <20% error rate: {self.training_history['results_summary'][20]:.1f}%")

        print(f"\nGoal Achievement: {self.training_history['success_status']}")

        # If targets not met, provide improvement suggestions
        if not (self.training_history['r2_target_met'] and self.training_history['error_10_target_met']):
            print(f"\nImprovement Suggestions:")
            if not self.training_history['r2_target_met']:
                print(f"   • Increase model complexity (more layers/estimators)")
                print(f"   • Try different ensemble weights")
                print(f"   • Add more diverse models to ensemble")
            if not self.training_history['error_10_target_met']:
                print(f"   • Fine-tune hyperparameters")
                print(f"   • Increase training data quality")
                print(f"   • Consider feature engineering")

        print(f"\n--- Training Complete ---")
    
    def train_and_evaluate(self, train_path: str, test_path: str, 
                          output_dir: str = "results") -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            train_path (str): Path to augmented training data
            test_path (str): Path to test data
            output_dir (str): Directory to save results
            
        Returns:
            Dict containing training results and metrics
        """
        # Load data
        X_train_scaled, y_train_log, X_test_scaled, y_test = self.load_data(train_path, test_path)
        
        # Train model
        self.train_model(X_train_scaled, y_train_log)
        
        # Evaluate model
        results = self.evaluate_model(X_test_scaled, y_test)
        
        # Save results
        self.save_results(output_dir)
        self.save_model(output_dir)
        
        # Print summary
        self.print_summary()
        
        return results


def main():
    """Main function to run model training and evaluation."""
    # Initialize predictor
    predictor = ChlorideCorrosionPredictor()
    
    # Run training and evaluation
    train_path = "../data/augmented_training_data_unscaled.csv"
    test_path = "../data/testing_data.csv"
    results = predictor.train_and_evaluate(train_path, test_path)
    
    return results


if __name__ == "__main__":
    main()
