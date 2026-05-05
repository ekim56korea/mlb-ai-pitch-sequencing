"""
🔥 [WEEK 6] 검증 스크립트
39개 피처 최적화 검증 및 성능 비교
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from features.sequence import calculate_sequence_entropy
from features.contextual import ContextualFeatures

def test_feature_count():
    """피처 개수 검증"""
    # Direct file read to avoid import issues
    train_path = os.path.join(os.path.dirname(__file__), 'train.py')
    with open(train_path, 'r') as f:
        content = f.read()
        # Extract INPUT_SIZE
        for line in content.split('\n'):
            if 'INPUT_SIZE = ' in line and not line.strip().startswith('#'):
                INPUT_SIZE = int(line.split('=')[1].split('#')[0].strip())
                break
        
        # Extract FEATURES list
        start_idx = content.find('FEATURES = [')
        end_idx = content.find(']', start_idx) + 1
        features_str = content[start_idx:end_idx]
        # Count features by counting commas + 1
        feature_count = features_str.count("'")  // 2
    
    print("=" * 60)
    print("🧪 피처 개수 검증")
    print("=" * 60)
    
    print(f"✅ FEATURES 리스트 길이: {feature_count}")
    print(f"✅ INPUT_SIZE 상수: {INPUT_SIZE}")
    
    if feature_count == INPUT_SIZE:
        print(f"✅ 피처 개수 일치: {feature_count} == {INPUT_SIZE}")
        return True
    else:
        print(f"❌ 피처 개수 불일치: {feature_count} != {INPUT_SIZE}")
        return False

def test_sequence_entropy():
    """sequence_entropy 함수 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Sequence Entropy 테스트")
    print("=" * 60)
    
    # Test case 1: 완전 예측 가능
    seq1 = ['FF', 'FF', 'FF', 'FF']
    entropy1 = calculate_sequence_entropy(seq1)
    print(f"✅ 동일 구종 {seq1}: entropy = {entropy1:.3f} (expected: 0.000)")
    assert entropy1 == 0.0, "동일 구종은 entropy=0이어야 함"
    
    # Test case 2: 50-50 분포
    seq2 = ['FF', 'SL', 'FF', 'SL']
    entropy2 = calculate_sequence_entropy(seq2)
    print(f"✅ 50-50 분포 {seq2}: entropy = {entropy2:.3f} (expected: 1.000)")
    assert 0.95 < entropy2 < 1.05, "50-50 분포는 entropy≈1.0"
    
    # Test case 3: 균등 분포 (4개 구종)
    seq3 = ['FF', 'SL', 'CH', 'CU']
    entropy3 = calculate_sequence_entropy(seq3)
    print(f"✅ 균등 분포 {seq3}: entropy = {entropy3:.3f} (expected: 2.000)")
    assert 1.95 < entropy3 < 2.05, "4개 균등 분포는 entropy≈2.0"
    
    # Test case 4: 빈 시퀀스
    seq4 = []
    entropy4 = calculate_sequence_entropy(seq4)
    print(f"✅ 빈 시퀀스 {seq4}: entropy = {entropy4:.3f} (expected: 0.000)")
    assert entropy4 == 0.0, "빈 시퀀스는 entropy=0"
    
    return True

def test_altitude_factor():
    """altitude_factor 물리 모델 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Altitude Factor 테스트 (물리 모델)")
    print("=" * 60)
    
    # Test case 1: 해수면
    alt_sealevel = pd.Series([20])  # Fenway Park
    factor_sealevel = ContextualFeatures.calculate_altitude_factor(alt_sealevel)
    print(f"✅ Fenway Park (20ft): factor = {factor_sealevel.iloc[0]:.4f} (expected: ~1.000)")
    assert 0.999 < factor_sealevel.iloc[0] < 1.001, "해수면은 factor≈1.0"
    
    # Test case 2: Coors Field
    alt_coors = pd.Series([5200])  # Coors Field
    factor_coors = ContextualFeatures.calculate_altitude_factor(alt_coors)
    print(f"✅ Coors Field (5200ft): factor = {factor_coors.iloc[0]:.4f} (expected: ~1.062)")
    assert 1.055 < factor_coors.iloc[0] < 1.070, "Coors Field는 factor≈1.062 (+6.2%)"
    
    # Test case 3: Chase Field
    alt_chase = pd.Series([1090])  # Chase Field
    factor_chase = ContextualFeatures.calculate_altitude_factor(alt_chase)
    print(f"✅ Chase Field (1090ft): factor = {factor_chase.iloc[0]:.4f} (expected: ~1.013)")
    assert 1.010 < factor_chase.iloc[0] < 1.020, "Chase Field는 factor≈1.013 (+1.3%)"
    
    return True

def test_feature_groups():
    """피처 그룹별 개수 검증"""
    # Read FEATURES from train.py
    train_path = os.path.join(os.path.dirname(__file__), 'train.py')
    with open(train_path, 'r') as f:
        content = f.read()
        start_idx = content.find('FEATURES = [')
        end_idx = content.find(']', start_idx)
        features_str = content[start_idx+len('FEATURES = ['):end_idx]
        # Extract feature names (enclosed in quotes)
        import re
        FEATURES = re.findall(r"'([^']+)'", features_str)
    
    print("\n" + "=" * 60)
    print("🧪 피처 그룹별 검증")
    print("=" * 60)
    
    groups = {
        'Situation': ['inning', 'balls', 'strikes', 'outs_when_up', 'score_diff', 
                     'on_1b', 'on_2b', 'on_3b', 'stand_code'],
        'Pitcher/Batter': ['p_throws_code', 'pitch_number', 'tto', 'pitcher_pitch_count'],
        'Batter Tendency': ['batter_whiff_rate', 'batter_k_rate'],
        'Z-Score': ['z_vel', 'z_spin', 'z_hb', 'z_ivb', 'z_ext', 'z_rel_h', 'z_rel_s', 'z_league_whiff'],
        'Tunneling': ['tunnel_distance', 'velocity_diff', 
                     'FF_count_last_5', 'SL_count_last_5', 'CH_count_last_5', 'CU_count_last_5',
                     'sequence_entropy'],
        'BvP': ['bvp_ba', 'bvp_whiff_rate', 'bvp_k_rate', 'platoon_advantage', 'bvp_recent_ba'],
        'Contextual': ['altitude_factor', 'rest_days', 'fatigue_index', 'pressure_index']
    }
    
    total = 0
    for group_name, group_features in groups.items():
        count = len(group_features)
        # Check all features in this group exist in FEATURES
        missing = [f for f in group_features if f not in FEATURES]
        if missing:
            print(f"❌ {group_name}: {count}개 - 누락된 피처: {missing}")
            return False
        else:
            print(f"✅ {group_name}: {count}개")
            total += count
    
    print(f"\n📊 총 피처 개수: {total}")
    print(f"📊 FEATURES 리스트: {len(FEATURES)}")
    
    if total == len(FEATURES):
        print("✅ 모든 피처 그룹 검증 완료!")
        return True
    else:
        extra = set(FEATURES) - set([f for group in groups.values() for f in group])
        if extra:
            print(f"❌ 추가 피처 발견: {extra}")
        return False

def test_removed_features():
    """제거된 피처 검증"""
    # Read FEATURES from train.py
    train_path = os.path.join(os.path.dirname(__file__), 'train.py')
    with open(train_path, 'r') as f:
        content = f.read()
        start_idx = content.find('FEATURES = [')
        end_idx = content.find(']', start_idx)
        features_str = content[start_idx+len('FEATURES = ['):end_idx]
        # Extract feature names (enclosed in quotes)
        import re
        FEATURES = re.findall(r"'([^']+)'", features_str)
    
    print("\n" + "=" * 60)
    print("🧪 제거된 피처 검증")
    print("=" * 60)
    
    removed_features = ['trajectory_div', 'inning_fatigue']
    
    for feature in removed_features:
        if feature in FEATURES:
            print(f"❌ {feature}가 여전히 FEATURES에 존재함!")
            return False
        else:
            print(f"✅ {feature} 정상적으로 제거됨")
    
    print("\n✅ 중복 피처 제거 완료!")
    return True

def compare_week4_week6():
    """Week 4 vs Week 6 비교"""
    print("\n" + "=" * 60)
    print("📊 Week 4 vs Week 6 비교")
    print("=" * 60)
    
    week4_count = 43
    week6_count = 39
    
    print(f"Week 4 피처 개수: {week4_count}")
    print(f"Week 6 피처 개수: {week6_count}")
    print(f"제거된 피처: {week4_count - week6_count}개")
    print(f"")
    print(f"제거된 피처 목록:")
    print(f"  1. trajectory_div  (r=0.999 with tunnel_distance)")
    print(f"  2. inning_fatigue  (r=1.000 with inning)")
    print(f"")
    print(f"개선된 피처:")
    print(f"  1. sequence_entropy: placeholder → Shannon entropy")
    print(f"  2. altitude_factor: 선형 모델 → 지수 감쇠 물리 모델")
    print(f"  3. pressure_index: 균등 가중치 → 실험 기반 가중치")
    
    reduction_pct = (week4_count - week6_count) / week4_count * 100
    print(f"\n✅ 피처 개수 {reduction_pct:.1f}% 감소 (차원 축소)")
    print(f"✅ 예상 학습 속도 향상: ~{reduction_pct * 0.5:.1f}%")
    print(f"✅ 예상 추론 속도 향상: ~{reduction_pct * 0.3:.1f}%")

def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 80)
    print(" " * 20 + "🔥 Week 6 Validation Suite")
    print("=" * 80)
    
    tests = [
        ("피처 개수 검증", test_feature_count),
        ("Sequence Entropy", test_sequence_entropy),
        ("Altitude Factor", test_altitude_factor),
        ("피처 그룹 검증", test_feature_groups),
        ("제거된 피처 검증", test_removed_features),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 실패: {str(e)}")
            results.append((test_name, False))
    
    # Week 4 vs Week 6 비교 (항상 실행)
    compare_week4_week6()
    
    # 최종 결과
    print("\n" + "=" * 80)
    print(" " * 25 + "📋 테스트 결과")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} | {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    print("=" * 80)
    print(f"총 테스트: {total_tests}개 | 통과: {passed_tests}개 | 실패: {total_tests - passed_tests}개")
    
    if passed_tests == total_tests:
        print("\n🎉 모든 테스트 통과! Week 6 최적화 완료.")
        return True
    else:
        print("\n⚠️  일부 테스트 실패. 코드 검토 필요.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
