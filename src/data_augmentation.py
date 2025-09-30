"""
Data Augmentation Module for Chloride Corrosion Prediction

This module implements safe data augmentation techniques to increase the training dataset
size while maintaining data integrity and preventing negative values in features.

Author: Research Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import os
from typing import Tuple, Optional

warnings.filterwarnings('ignore')


class DataAugmenter:
    """
    A class for safely augmenting chloride corrosion dataset.
    
    This class implements percentage-based noise augmentation that guarantees
    no negative values for originally positive features.
    """
    
    def __init__(self, target_samples: int = 800):
        """
        Initialize the DataAugmenter.
        
        Args:
            target_samples (int): Target number of samples after augmentation
        """
        self.target_samples = target_samples
        self.feature_columns = [
            'Cover_Thickness_mm', 'Reinforcement_Diameter_mm', 'Water_Cement_Ratio',
            'Temperature_K', 'Relative_Humidity_pct', 'Chloride_Ion_Content_kgm3', 
            'Time_Years'
        ]
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load and prepare the chloride corrosion dataset.
        
        Args:
            data_path (str): Path to the CSV file
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        print("--- Loading Data ---")
        try:
            df = pd.read_csv(data_path)
            X = df.drop('Corrosion_Rate_uAcm2', axis=1)
            y = df['Corrosion_Rate_uAcm2'] + 1e-6  # Prevent log(0)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"Original training samples: {len(X_train)}")
            print(f"Features: {list(X.columns)}")
            return X_train, X_test, y_train, y_test
        except FileNotFoundError:
            print(f"Error: CSV file not found at {data_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def safe_augment_data(self, X_original: pd.DataFrame, y_log: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform safe data augmentation that guarantees no negative values.
        
        Uses percentage-based noise and strict positive constraints.
        
        Args:
            X_original (pd.DataFrame): Original feature data
            y_log (pd.Series): Log-transformed target values
            
        Returns:
            Tuple of (X_augmented, y_augmented) as numpy arrays
        """
        print(f"\n--- Starting Safe Data Augmentation ---")
        print(f"Target: ~{self.target_samples} total samples")

        # Calculate how many augmentation rounds we need
        original_count = len(X_original)
        augmentation_rounds = int((self.target_samples - original_count) / original_count)

        print(f"Original samples: {original_count}")
        print(f"Augmentation rounds: {augmentation_rounds}")

        # Start with original data
        X_augmented_list = [X_original.values.copy()]
        y_augmented_list = [y_log.values.copy()]

        # Get original data statistics for each feature
        for feature_idx, feature_name in enumerate(X_original.columns):
            feature_data = X_original.iloc[:, feature_idx]
            print(f"Feature '{feature_name}': min={feature_data.min():.4f}, max={feature_data.max():.4f}")

        # Perform augmentation rounds
        for round_num in range(augmentation_rounds):
            print(f"\nAugmentation round {round_num + 1}/{augmentation_rounds}")

            # Create augmented version of original data
            X_new = X_original.values.copy()

            # Apply noise to each feature individually with strict constraints
            for feature_idx in range(X_original.shape[1]):
                feature_data = X_original.iloc[:, feature_idx].values
                feature_name = X_original.columns[feature_idx]

                # Calculate feature statistics
                feature_min = feature_data.min()
                feature_max = feature_data.max()
                feature_std = feature_data.std()

                # Use very small percentage-based noise (1-2% of the value)
                noise_percentage = 0.01 + (round_num * 0.005)  # Gradually increase noise each round

                # Generate noise as percentage of each individual value
                relative_noise = np.random.normal(0, noise_percentage, size=len(feature_data))

                # Apply noise
                noisy_feature = feature_data * (1 + relative_noise)

                # Ensure no negative values for originally positive features
                if feature_min >= 0:  # Feature should stay non-negative
                    # Method 1: Absolute value for any negatives
                    noisy_feature = np.abs(noisy_feature)

                    # Method 2: Additional safety - clip to reasonable minimum
                    min_allowed = max(0.001, feature_min * 0.1)  # At least 10% of original minimum
                    noisy_feature = np.maximum(noisy_feature, min_allowed)

                    # Method 3: Don't let it exceed reasonable maximum
                    max_allowed = feature_max * 2.0  # Don't exceed 2x original maximum
                    noisy_feature = np.minimum(noisy_feature, max_allowed)

                else:  # Feature can be negative, just apply reasonable bounds
                    # Don't let it go too far from original range
                    lower_bound = feature_min - feature_std
                    upper_bound = feature_max + feature_std
                    noisy_feature = np.clip(noisy_feature, lower_bound, upper_bound)

                # Update the feature in new data
                X_new[:, feature_idx] = noisy_feature

                # Verify no negatives for this feature
                negative_count = np.sum(X_new[:, feature_idx] < 0)
                if negative_count > 0 and feature_min >= 0:
                    print(f"WARNING: Feature '{feature_name}' has {negative_count} negative values!")
                    # Emergency fix - replace with small positive values
                    negative_mask = X_new[:, feature_idx] < 0
                    X_new[negative_mask, feature_idx] = np.random.uniform(
                        0.001, feature_min * 0.5, size=np.sum(negative_mask)
                    )

            # Add this round's data to our lists
            X_augmented_list.append(X_new.copy())
            y_augmented_list.append(y_log.values.copy())

            # Check for negative values in this round
            round_negatives = np.sum(X_new < 0)
            print(f"Round {round_num + 1}: {round_negatives} negative values")

        # Combine all augmented data
        X_final = np.vstack(X_augmented_list)
        y_final = np.concatenate(y_augmented_list)

        # Final verification
        total_negatives = np.sum(X_final < 0)
        print(f"\n--- FINAL VERIFICATION ---")
        print(f"Total samples after augmentation: {len(X_final)}")
        print(f"Total negative values: {total_negatives}")

        if total_negatives > 0:
            print("ERROR: Still have negative values! Let's fix them...")
            # Emergency cleanup - replace ALL negative values
            for feature_idx in range(X_final.shape[1]):
                feature_col = X_final[:, feature_idx]
                if np.any(feature_col < 0):
                    original_feature = X_original.iloc[:, feature_idx].values
                    feature_min = original_feature.min()

                    if feature_min >= 0:  # Should be non-negative
                        negative_mask = feature_col < 0
                        # Replace with small random positive values
                        replacement_values = np.random.uniform(
                            0.001, feature_min + 0.1, size=np.sum(negative_mask)
                        )
                        X_final[negative_mask, feature_idx] = replacement_values
                        print(f"Fixed {np.sum(negative_mask)} negative values in feature {feature_idx}")

        # Re-verify
        final_negatives = np.sum(X_final < 0)
        print(f"After cleanup: {final_negatives} negative values")

        # Show statistics for each feature
        print(f"\n--- Feature Statistics After Augmentation ---")
        for feature_idx, feature_name in enumerate(X_original.columns):
            original_min = X_original.iloc[:, feature_idx].min()
            original_max = X_original.iloc[:, feature_idx].max()
            augmented_min = X_final[:, feature_idx].min()
            augmented_max = X_final[:, feature_idx].max()

            print(f"{feature_name}:")
            print(f"  Original: [{original_min:.4f}, {original_max:.4f}]")
            print(f"  Augmented: [{augmented_min:.4f}, {augmented_max:.4f}]")
            if augmented_min < 0 and original_min >= 0:
                print(f"  WARNING: Negative values in originally positive feature!")

        return X_final, y_final
    
    def save_augmented_data(self, X_augmented: np.ndarray, y_augmented: np.ndarray, 
                           X_original: pd.DataFrame, y_original: pd.Series, 
                           output_dir: str = "data") -> None:
        """
        Save augmented and original data to CSV files.
        
        Args:
            X_augmented (np.ndarray): Augmented feature data
            y_augmented (np.ndarray): Augmented target data
            X_original (pd.DataFrame): Original feature data
            y_original (pd.Series): Original target data
            output_dir (str): Directory to save files
        """
        print(f"\n--- Saving Results ---")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save augmented data (unscaled)
        augmented_df = pd.DataFrame(X_augmented, columns=self.feature_columns)
        augmented_df['log_Corrosion_Rate_uAcm2'] = y_augmented
        augmented_path = os.path.join(output_dir, 'augmented_training_data_unscaled.csv')
        augmented_df.to_csv(augmented_path, index=False)

        # Save original data for comparison
        original_df = pd.DataFrame(X_original.values, columns=self.feature_columns)
        original_df['log_Corrosion_Rate_uAcm2'] = y_original.values
        original_path = os.path.join(output_dir, 'original_training_data.csv')
        original_df.to_csv(original_path, index=False)

        print(f"Augmented data saved to '{augmented_path}'")
        print(f"Original data saved to '{original_path}'")
    
    def augment_dataset(self, data_path: str, output_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete data augmentation pipeline.
        
        Args:
            data_path (str): Path to the input CSV file
            output_dir (str): Directory to save output files
            
        Returns:
            Tuple of (X_augmented, y_augmented) as numpy arrays
        """
        # Load data
        X_train, X_test, y_train, y_test = self.load_data(data_path)
        
        # Log transform target
        y_train_log = np.log1p(y_train)
        
        # Apply augmentation
        X_train_augmented, y_train_log_augmented = self.safe_augment_data(X_train, y_train_log)
        
        # Save results
        self.save_augmented_data(X_train_augmented, y_train_log_augmented, 
                                X_train, y_train_log, output_dir)
        
        # Final summary
        print(f"\n--- FINAL SUMMARY ---")
        print(f"Original samples: {len(X_train)}")
        print(f"Augmented samples: {len(X_train_augmented)}")
        print(f"Augmentation ratio: {len(X_train_augmented)/len(X_train):.1f}x")
        print(f"Negative values: {np.sum(X_train_augmented < 0)}")

        if np.sum(X_train_augmented < 0) == 0:
            print(f"SUCCESS: No negative values in augmented data!")
        else:
            print(f"FAILED: Still have negative values")
        
        return X_train_augmented, y_train_log_augmented


def main():
    """Main function to run data augmentation."""
    # Initialize augmenter
    augmenter = DataAugmenter(target_samples=800)
    
    # Run augmentation
    data_path = "../data/Chloride_dataset.csv"  # Adjust path as needed
    X_augmented, y_augmented = augmenter.augment_dataset(data_path)
    
    return X_augmented, y_augmented


if __name__ == "__main__":
    main()
