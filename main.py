"""
Main Pipeline for Chloride Corrosion Prediction

This script orchestrates the complete machine learning pipeline for chloride corrosion
prediction, including data augmentation, model training, and sensitivity analysis.

Author: Research Assistant
Date: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_augmentation import DataAugmenter
from model_training import ChlorideCorrosionPredictor
from sensitivity_analysis import SensitivityAnalyzer


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chloride_corrosion.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = ['data', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def run_data_augmentation(input_data_path: str, output_dir: str = "data") -> tuple:
    """
    Run data augmentation pipeline.
    
    Args:
        input_data_path (str): Path to input CSV file
        output_dir (str): Directory to save augmented data
        
    Returns:
        tuple: (X_augmented, y_augmented)
    """
    print("=" * 60)
    print("STEP 1: DATA AUGMENTATION")
    print("=" * 60)
    
    augmenter = DataAugmenter(target_samples=800)
    X_augmented, y_augmented = augmenter.augment_dataset(input_data_path, output_dir)
    
    return X_augmented, y_augmented


def run_model_training(train_path: str, test_path: str, output_dir: str = "results") -> dict:
    """
    Run model training and evaluation pipeline.
    
    Args:
        train_path (str): Path to augmented training data
        test_path (str): Path to test data
        output_dir (str): Directory to save results
        
    Returns:
        dict: Training results and metrics
    """
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    predictor = ChlorideCorrosionPredictor()
    results = predictor.train_and_evaluate(train_path, test_path, output_dir)
    
    return results


def run_sensitivity_analysis(train_path: str, test_path: str, 
                           model_path: str = None, scaler_path: str = None,
                           output_dir: str = "results") -> tuple:
    """
    Run sensitivity analysis pipeline.
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
        model_path (str): Path to saved model (optional)
        scaler_path (str): Path to saved scaler (optional)
        output_dir (str): Directory to save results
        
    Returns:
        tuple: (sensitivity_df, detailed_df)
    """
    print("\n" + "=" * 60)
    print("STEP 3: SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    analyzer = SensitivityAnalyzer()
    sensitivity_df, detailed_df = analyzer.run_complete_analysis(
        train_path, test_path, model_path, scaler_path, output_dir
    )
    
    return sensitivity_df, detailed_df


def run_complete_pipeline(input_data_path: str, test_data_path: str, 
                         output_dir: str = "results", skip_augmentation: bool = False) -> dict:
    """
    Run the complete chloride corrosion prediction pipeline.
    
    Args:
        input_data_path (str): Path to input CSV file
        test_data_path (str): Path to test CSV file
        output_dir (str): Directory to save all results
        skip_augmentation (bool): Whether to skip data augmentation
        
    Returns:
        dict: Complete pipeline results
    """
    print("CHLORIDE CORROSION PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Input data: {input_data_path}")
    print(f"Test data: {test_data_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    pipeline_results = {}
    
    # Step 1: Data Augmentation (optional)
    if not skip_augmentation:
        try:
            X_augmented, y_augmented = run_data_augmentation(input_data_path, "data")
            pipeline_results['augmentation'] = {
                'X_augmented': X_augmented,
                'y_augmented': y_augmented,
                'status': 'completed'
            }
            train_data_path = "data/augmented_training_data_unscaled.csv"
        except Exception as e:
            print(f"Data augmentation failed: {str(e)}")
            print("Proceeding with original data...")
            train_data_path = input_data_path
            pipeline_results['augmentation'] = {'status': 'failed', 'error': str(e)}
    else:
        print("Skipping data augmentation...")
        train_data_path = input_data_path
        pipeline_results['augmentation'] = {'status': 'skipped'}
    
    # Step 2: Model Training
    try:
        training_results = run_model_training(train_data_path, test_data_path, output_dir)
        pipeline_results['training'] = training_results
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        pipeline_results['training'] = {'status': 'failed', 'error': str(e)}
        return pipeline_results
    
    # Step 3: Sensitivity Analysis
    try:
        model_path = os.path.join(output_dir, 'chloride_corrosion_model.pkl')
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        
        sensitivity_df, detailed_df = run_sensitivity_analysis(
            train_data_path, test_data_path, model_path, scaler_path, output_dir
        )
        pipeline_results['sensitivity'] = {
            'sensitivity_df': sensitivity_df,
            'detailed_df': detailed_df,
            'status': 'completed'
        }
    except Exception as e:
        print(f"Sensitivity analysis failed: {str(e)}")
        pipeline_results['sensitivity'] = {'status': 'failed', 'error': str(e)}
    
    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETION SUMMARY")
    print("=" * 60)
    
    for step, result in pipeline_results.items():
        status = result.get('status', 'unknown')
        print(f"{step.upper()}: {status}")
        if status == 'failed':
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Pipeline completed!")
    
    return pipeline_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Chloride Corrosion Prediction Pipeline')
    
    parser.add_argument('--input-data', type=str, required=True,
                       help='Path to input CSV file (Chloride_dataset.csv)')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test CSV file (testing_data.csv)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--skip-augmentation', action='store_true',
                       help='Skip data augmentation step')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--step', type=str, choices=['augmentation', 'training', 'sensitivity', 'all'],
                       default='all', help='Run specific step only (default: all)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input files
    if not os.path.exists(args.input_data):
        logger.error(f"Input data file not found: {args.input_data}")
        sys.exit(1)
    
    if not os.path.exists(args.test_data):
        logger.error(f"Test data file not found: {args.test_data}")
        sys.exit(1)
    
    try:
        if args.step == 'all':
            # Run complete pipeline
            results = run_complete_pipeline(
                args.input_data, args.test_data, args.output_dir, args.skip_augmentation
            )
        elif args.step == 'augmentation':
            # Run only data augmentation
            X_aug, y_aug = run_data_augmentation(args.input_data, "data")
            results = {'augmentation': {'X_augmented': X_aug, 'y_augmented': y_aug}}
        elif args.step == 'training':
            # Run only model training
            train_path = "data/augmented_training_data_unscaled.csv"
            if not os.path.exists(train_path):
                logger.error(f"Augmented training data not found: {train_path}")
                logger.error("Run augmentation step first or use --skip-augmentation")
                sys.exit(1)
            results = run_model_training(train_path, args.test_data, args.output_dir)
        elif args.step == 'sensitivity':
            # Run only sensitivity analysis
            train_path = "data/augmented_training_data_unscaled.csv"
            model_path = os.path.join(args.output_dir, 'chloride_corrosion_model.pkl')
            scaler_path = os.path.join(args.output_dir, 'feature_scaler.pkl')
            
            if not os.path.exists(train_path):
                logger.error(f"Training data not found: {train_path}")
                sys.exit(1)
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                logger.error("Run training step first")
                sys.exit(1)
            
            sensitivity_df, detailed_df = run_sensitivity_analysis(
                train_path, args.test_data, model_path, scaler_path, args.output_dir
            )
            results = {'sensitivity': {'sensitivity_df': sensitivity_df, 'detailed_df': detailed_df}}
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
