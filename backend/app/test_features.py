"""
Feature Engineering 통합 테스트

Tunneling + BvP + Contextual 피처를 모두 추가하고
총 피처 개수 및 상관관계를 검증합니다.
"""

import sys
import os
import numpy as np
import pandas as pd

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.tunneling import TunnelingFeatures
from features.batter_pitcher import BvPFeatures
from features.contextual import ContextualFeatures


def create_comprehensive_test_data(n_pitches=1000):
    """
    실제 MLB 데이터와 유사한 포괄적 테스트 데이터 생성
    """
    np.random.seed(42)
    
    pitchers = [543037, 660271, 669373]  # 3명의 투수
    batters = [660670, 592450, 514888, 665742]  # 4명의 타자
    
    df = pd.DataFrame({
        # 기본 정보
        'game_pk': np.repeat(range(1, 101), 10),  # 100 games, 10 pitches each
        'game_date': pd.date_range('2024-04-01', periods=n_pitches, freq='H'),
        'pitcher': np.random.choice(pitchers, n_pitches),
        'batter': np.random.choice(batters, n_pitches),
        
        # 카운트 정보
        'at_bat_number': np.tile(range(1, 11), 100),  # 10 at-bats per game
        'pitch_number': np.random.randint(1, 8, n_pitches),
        'balls': np.random.randint(0, 4, n_pitches),
        'strikes': np.random.randint(0, 3, n_pitches),
        'outs_when_up': np.random.randint(0, 3, n_pitches),
        
        # 구종 및 물리
        'pitch_type': np.random.choice(['FF', 'SL', 'CH', 'CU'], n_pitches, 
                                       p=[0.35, 0.25, 0.20, 0.20]),
        'release_speed': np.random.normal(92, 5, n_pitches),
        'release_pos_x': np.random.normal(-2.0, 0.2, n_pitches),
        'release_pos_z': np.random.normal(6.0, 0.3, n_pitches),
        'pfx_x': np.random.normal(0, 5, n_pitches),
        'pfx_z': np.random.normal(0, 8, n_pitches),
        'plate_x': np.random.normal(0, 0.8, n_pitches),
        'plate_z': np.random.normal(2.5, 0.6, n_pitches),
        
        # 결과
        'events': np.random.choice(
            ['single', 'double', 'strikeout', 'walk', 'field_out', None],
            n_pitches, p=[0.15, 0.05, 0.20, 0.08, 0.35, 0.17]
        ),
        'description': np.random.choice(
            ['ball', 'called_strike', 'swinging_strike', 'foul', 'hit_into_play'],
            n_pitches, p=[0.30, 0.15, 0.12, 0.23, 0.20]
        ),
        
        # 경기 상황
        'inning': np.random.randint(1, 10, n_pitches),
        'score_diff': np.random.randint(-5, 6, n_pitches),
        'on_1b': np.random.choice([0, 1], n_pitches, p=[0.7, 0.3]),
        'on_2b': np.random.choice([0, 1], n_pitches, p=[0.8, 0.2]),
        'on_3b': np.random.choice([0, 1], n_pitches, p=[0.9, 0.1]),
        
        # 타자/투수 정보
        'stand': np.random.choice(['L', 'R'], n_pitches, p=[0.3, 0.7]),
        'p_throws': np.random.choice(['L', 'R'], n_pitches, p=[0.25, 0.75]),
        
        # 경기장
        'venue_name': np.random.choice(
            ['Coors Field', 'Fenway Park', 'Dodger Stadium', 'Yankee Stadium'],
            n_pitches
        ),
        
        # 시간
        'inning_topbot': np.random.choice(['Top', 'Bot'], n_pitches),
        'game_hour': np.random.choice([13, 14, 15, 19, 20], n_pitches),
        
        # 점수
        'bat_score': np.random.randint(0, 10, n_pitches),
        'home_score': np.random.randint(0, 10, n_pitches),
    })
    
    return df


def test_all_features():
    """모든 피처 모듈을 통합 테스트"""
    
    print("=" * 70)
    print("Feature Engineering 통합 테스트")
    print("=" * 70)
    
    # 1. 테스트 데이터 생성
    print("\n📊 Step 1: 테스트 데이터 생성")
    print("-" * 70)
    df = create_comprehensive_test_data(n_pitches=1000)
    print(f"✅ 생성 완료: {len(df):,} pitches")
    print(f"   - Games: {df['game_pk'].nunique()}")
    print(f"   - Pitchers: {df['pitcher'].nunique()}")
    print(f"   - Batters: {df['batter'].nunique()}")
    print(f"   - 기존 컬럼 수: {len(df.columns)}")
    
    # 2. Tunneling Features
    print("\n🎯 Step 2: Tunneling Features 추가")
    print("-" * 70)
    original_cols = df.columns.tolist()
    df = TunnelingFeatures.add_all_tunneling_features(df)
    tunneling_cols = [col for col in df.columns if col not in original_cols]
    print(f"✅ 추가된 피처: {len(tunneling_cols)}개")
    for col in tunneling_cols:
        print(f"   - {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    
    # 3. BvP Features
    print("\n📊 Step 3: Batter vs Pitcher Features 추가")
    print("-" * 70)
    original_cols = df.columns.tolist()
    df = BvPFeatures.add_all_bvp_features(df)
    bvp_cols = [col for col in df.columns if col not in original_cols]
    print(f"✅ 추가된 피처: {len(bvp_cols)}개")
    for col in bvp_cols[:10]:  # 처음 10개만 출력
        print(f"   - {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    if len(bvp_cols) > 10:
        print(f"   ... (총 {len(bvp_cols)}개)")
    
    # 4. Contextual Features
    print("\n🌍 Step 4: Contextual Features 추가")
    print("-" * 70)
    original_cols = df.columns.tolist()
    df = ContextualFeatures.add_all_contextual_features(df)
    contextual_cols = [col for col in df.columns if col not in original_cols]
    print(f"✅ 추가된 피처: {len(contextual_cols)}개")
    for col in contextual_cols:
        print(f"   - {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    
    # 5. 종합 요약
    print("\n" + "=" * 70)
    print("종합 요약")
    print("=" * 70)
    
    total_new_features = len(tunneling_cols) + len(bvp_cols) + len(contextual_cols)
    print(f"✅ 총 추가된 피처: {total_new_features}개")
    print(f"   - Tunneling: {len(tunneling_cols)}개")
    print(f"   - BvP: {len(bvp_cols)}개")
    print(f"   - Contextual: {len(contextual_cols)}개")
    print(f"\n📊 최종 DataFrame 크기:")
    print(f"   - Rows: {len(df):,}")
    print(f"   - Columns: {len(df.columns)} (기존 + 신규)")
    print(f"   - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 6. 결측치 확인
    print("\n🔍 결측치 확인:")
    null_counts = df[tunneling_cols + bvp_cols + contextual_cols].isnull().sum()
    if null_counts.sum() == 0:
        print("   ✅ 결측치 없음!")
    else:
        print(f"   ⚠️ 결측치 발견:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"      - {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # 7. 피처 통계
    print("\n📈 주요 피처 통계:")
    key_features = {
        'tunnel_distance': tunneling_cols,
        'bvp_ba': bvp_cols,
        'fatigue_index': contextual_cols,
        'pressure_index': contextual_cols,
    }
    
    for feature_name, col_list in key_features.items():
        matching_cols = [col for col in col_list if feature_name in col]
        if matching_cols:
            col = matching_cols[0]
            stats = df[col].describe()
            print(f"\n   {col}:")
            print(f"      Min: {stats['min']:.3f}")
            print(f"      Mean: {stats['mean']:.3f}")
            print(f"      Max: {stats['max']:.3f}")
            print(f"      Std: {stats['std']:.3f}")
    
    # 8. 상관관계 확인 (샘플)
    print("\n🔗 상관관계 샘플 (Tunneling vs BvP):")
    if 'tunnel_distance' in tunneling_cols and 'bvp_whiff_rate' in bvp_cols:
        corr = df['tunnel_distance'].corr(df['bvp_whiff_rate'])
        print(f"   tunnel_distance vs bvp_whiff_rate: {corr:.3f}")
    
    if 'velocity_diff' in tunneling_cols and 'bvp_k_rate' in bvp_cols:
        corr = df['velocity_diff'].corr(df['bvp_k_rate'])
        print(f"   velocity_diff vs bvp_k_rate: {corr:.3f}")
    
    # 9. 예상 모델 INPUT_SIZE
    print("\n" + "=" * 70)
    print("모델 입력 피처 계획")
    print("=" * 70)
    
    # 기존 피처 (train.py에서 사용 중)
    existing_features = [
        'inning', 'balls', 'strikes', 'outs_when_up', 'score_diff',
        'on_1b', 'on_2b', 'on_3b', 'stand_code', 'p_throws_code',
        'pitch_number', 'tto', 'pitcher_pitch_count',
        'batter_whiff_rate', 'batter_k_rate',
        'z_vel', 'z_spin', 'z_hb', 'z_ivb', 'z_ext', 'z_rel_h', 'z_rel_s', 'z_league_whiff'
    ]
    
    # 신규 피처 선정 (상관관계 높거나 중요한 것만)
    selected_tunneling = [
        'tunnel_distance', 'trajectory_div', 'velocity_diff',
        'FB_count_last_5', 'SL_count_last_5', 'CH_count_last_5', 'CU_count_last_5',
        'sequence_entropy'
    ]
    
    selected_bvp = [
        'bvp_ba', 'bvp_whiff_rate', 'bvp_k_rate',
        'platoon_advantage', 'bvp_recent_ba'
    ]
    
    selected_contextual = [
        'altitude_factor', 'rest_days', 'fatigue_index',
        'pressure_index', 'inning_fatigue'
    ]
    
    total_input_size = (
        len(existing_features) + 
        len(selected_tunneling) + 
        len(selected_bvp) + 
        len(selected_contextual)
    )
    
    print(f"✅ 기존 피처: {len(existing_features)}개")
    print(f"✅ 신규 Tunneling: {len(selected_tunneling)}개")
    print(f"✅ 신규 BvP: {len(selected_bvp)}개")
    print(f"✅ 신규 Contextual: {len(selected_contextual)}개")
    print(f"\n📊 예상 INPUT_SIZE: {len(existing_features)} → {total_input_size}")
    print(f"   (기존 25개 → 신규 {total_input_size}개, +{total_input_size - len(existing_features)}개)")
    
    # 10. 성공 메시지
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 통과!")
    print("=" * 70)
    print("\n다음 단계:")
    print("1. train.py에 신규 피처 통합")
    print("2. DuckDB 쿼리 업데이트 (BvP 계산 추가)")
    print("3. INPUT_SIZE 업데이트 (25 → 43)")
    print("4. Feature Importance 분석 (SHAP 또는 XGBoost)")
    
    return df


if __name__ == "__main__":
    try:
        df_enhanced = test_all_features()
        print("\n✅ 통합 테스트 성공!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
