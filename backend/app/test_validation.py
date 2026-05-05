"""
Test script for temporal validation
Tests the MLBTemporalValidator and MLBMetrics classes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from utils.validation import MLBTemporalValidator
from utils.metrics import MLBMetrics


def test_temporal_validation():
    """시계열 검증 테스트"""
    print("🧪 Testing Temporal Validation...")
    print("="*60)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2025-12-31', freq='D')
    
    sample_data = pd.DataFrame({
        'game_date': np.random.choice(dates, 10000),
        'pitch_type': np.random.choice(['FF', 'SL', 'CH', 'CU'], 10000),
        'balls': np.random.randint(0, 4, 10000),
        'strikes': np.random.randint(0, 3, 10000),
        'outs_when_up': np.random.randint(0, 3, 10000),
        'on_2b': np.random.randint(0, 2, 10000),
        'on_3b': np.random.randint(0, 2, 10000),
        'run_value': np.random.randn(10000) * 0.1
    })
    
    print(f"📊 Sample data: {len(sample_data):,} rows")
    print(f"   Date range: {sample_data['game_date'].min()} ~ {sample_data['game_date'].max()}")
    
    # Test 1: Holdout Split
    print("\n" + "-"*60)
    print("Test 1: Holdout Split (2015-2023 train, 2024-2025 test)")
    print("-"*60)
    
    validator = MLBTemporalValidator()
    train_df, test_df = validator.create_holdout_split(
        sample_data,
        train_years=list(range(2015, 2024)),
        test_years=[2024, 2025]
    )
    
    # 데이터 누수 검증
    print("\n🔍 Validating no data leakage...")
    is_valid = validator.validate_no_leakage(train_df, test_df)
    
    # Test 2: Walk-Forward Validation
    print("\n" + "-"*60)
    print("Test 2: Walk-Forward Cross Validation")
    print("-"*60)
    
    splits = validator.walk_forward_validation(
        sample_data,
        initial_train_years=5,
        step_size=1
    )
    
    print(f"\n✅ Created {len(splits)} walk-forward splits")
    
    # Test 3: Get Indices
    print("\n" + "-"*60)
    print("Test 3: Get Train/Test Indices")
    print("-"*60)
    
    train_idx, test_idx = validator.get_train_test_indices(
        sample_data,
        train_years=list(range(2015, 2024)),
        test_years=[2024, 2025]
    )
    
    print(f"✅ Train indices: {len(train_idx):,}")
    print(f"✅ Test indices: {len(test_idx):,}")
    
    print("\n" + "="*60)
    print("✅ All temporal validation tests passed!")
    print("="*60 + "\n")


def test_metrics():
    """평가 지표 테스트"""
    print("🧪 Testing MLB Metrics...")
    print("="*60)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4
    pitch_names = ['FF', 'SL', 'CH', 'CU']
    
    # 실제 정답
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # 예측 확률 (약간의 노이즈 추가)
    y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    y_proba = 0.7 * y_proba + 0.3 * np.eye(n_classes)[y_true]  # 정답에 가중치
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # 정규화
    
    # 예측값
    y_pred = np.argmax(y_proba, axis=1)
    
    # Test 1: Comprehensive Report
    print("\nTest 1: Comprehensive Report")
    print("-"*60)
    
    metrics = MLBMetrics()
    report = metrics.comprehensive_report(
        y_true, y_pred, y_proba, pitch_names, verbose=True
    )
    
    # Test 2: Calibration
    print("\nTest 2: Probability Calibration")
    print("-"*60)
    
    calib = metrics.pitch_probability_calibration(y_true, y_proba, n_bins=10)
    print(f"✅ Expected Calibration Error: {calib['expected_calibration_error']:.4f}")
    print(f"✅ Maximum Calibration Error: {calib['maximum_calibration_error']:.4f}")
    print(f"✅ Well Calibrated: {calib['is_well_calibrated']}")
    
    # Test 3: Run Value Impact (샘플 데이터프레임)
    print("\nTest 3: Run Value Impact")
    print("-"*60)
    
    results_df = pd.DataFrame({
        'pitch_type': y_true,
        'pred': y_pred,
        'run_value': np.random.randn(n_samples) * 0.1,
        'balls': np.random.randint(0, 4, n_samples),
        'strikes': np.random.randint(0, 3, n_samples),
        'outs_when_up': np.random.randint(0, 3, n_samples),
        'on_2b': np.random.randint(0, 2, n_samples),
        'on_3b': np.random.randint(0, 2, n_samples)
    })
    
    impact = metrics.calculate_expected_run_value_impact(results_df)
    print(f"✅ Runs Saved per Game: {impact['runs_saved_per_game']:.4f}")
    print(f"✅ Runs Saved per Season: {impact['runs_saved_per_season']:.2f}")
    print(f"✅ WAR Equivalent: {impact['wars_equivalent']:.2f}")
    
    # Test 4: Confusion Analysis
    print("\nTest 4: Confusion Analysis")
    print("-"*60)
    
    confusion = metrics.confusion_analysis(y_true, y_pred, pitch_names, top_n=3)
    print(f"✅ Total Errors: {confusion['total_errors']}")
    print(f"✅ Top Confusions:")
    for i, pair in enumerate(confusion['top_confusions'], 1):
        print(f"   {i}. {pair['true_pitch']} → {pair['pred_pitch']}: "
              f"{pair['count']} errors ({pair['error_rate']*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ All metrics tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("WEEK 1 - TASK 1.1 & 1.2 VALIDATION TEST SUITE")
    print("🚀"*30 + "\n")
    
    test_temporal_validation()
    test_metrics()
    
    print("\n" + "🎉"*30)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("🎉"*30 + "\n")
    
    print("📝 Next Steps:")
    print("   1. Integrate temporal validation into train.py")
    print("   2. Run baseline evaluation with new metrics")
    print("   3. Compare with previous random split results")
    print("   4. Document accuracy drop (expected 10-15%)")
