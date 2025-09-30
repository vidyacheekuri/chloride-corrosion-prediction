"""
Test script for the Chloride Corrosion Prediction pipeline.

This script provides a simple way to test the pipeline functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import RobustScaler
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import xgboost as xgb
        import matplotlib.pyplot as plt
        import seaborn as sns
        import joblib
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def test_data_files():
    """Test if required data files exist."""
    print("Testing data files...")
    
    data_dir = Path("data")
    required_files = [
        "Chloride_dataset.csv",
        "testing_data.csv"
    ]
    
    all_exist = True
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    return all_exist

def test_pipeline_modules():
    """Test if pipeline modules can be imported."""
    print("Testing pipeline modules...")
    
    try:
        sys.path.append('src')
        from data_augmentation import DataAugmenter
        from model_training import ChlorideCorrosionPredictor
        from sensitivity_analysis import SensitivityAnalyzer
        print("✓ All pipeline modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Module import error: {e}")
        return False

def run_quick_test():
    """Run a quick test of the pipeline."""
    print("Running quick pipeline test...")
    
    try:
        # Test data augmentation
        sys.path.append('src')
        from data_augmentation import DataAugmenter
        
        augmenter = DataAugmenter(target_samples=100)  # Small test
        print("✓ DataAugmenter initialized")
        
        # Test model training
        from model_training import ChlorideCorrosionPredictor
        predictor = ChlorideCorrosionPredictor()
        print("✓ ChlorideCorrosionPredictor initialized")
        
        # Test sensitivity analysis
        from sensitivity_analysis import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer()
        print("✓ SensitivityAnalyzer initialized")
        
        print("✓ All modules initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("CHLORIDE CORROSION PREDICTION - TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Files Test", test_data_files),
        ("Module Import Test", test_pipeline_modules),
        ("Quick Pipeline Test", run_quick_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("\nTo run the complete pipeline:")
        print("python main.py --input-data data/Chloride_dataset.csv --test-data data/testing_data.csv")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Make sure to install requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
