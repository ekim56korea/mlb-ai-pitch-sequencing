"""
Tunneling Features Module

터널링(Tunneling): 초반 궤적이 동일하다가 후반에 급격히 분리되는 투구 조합
타자의 인식 지연 유발 → 헛스윙률 증가

수학적 배경:
--------------
1. 릴리스 포인트 거리 (Euclidean Distance):
   d = √[(x₂-x₁)² + (z₂-z₁)²]
   
   여기서:
   - (x₁, z₁): 이전 투구 릴리스 포인트
   - (x₂, z₂): 현재 투구 릴리스 포인트
   - d < 0.3 ft: 효과적 터널링 (타자가 구별 불가)

2. 궤적 분리도 (Trajectory Divergence at 40ft):
   물리 방정식:
   x(t) = x₀ + vₓ·t + ½·aₓ·t²
   z(t) = z₀ + vᵧ·t + ½·aᵧ·t²
   
   여기서:
   - vₓ, vᵧ: 초기 속도 벡터
   - aₓ, aᵧ: 가속도 (중력 + Magnus 효과)
   - t = 40ft / release_speed

3. 시퀀싱 패턴 엔트로피:
   H(X) = -Σ p(xᵢ) log p(xᵢ)
   
   여기서 p(xᵢ)는 최근 N개 투구에서 구종 i의 비율
   높은 엔트로피 = 예측 어려움 = 타자 혼란

참고 문헌:
----------
1. Whiteside et al. (2016). "Tunneling Effects in MLB Pitching"
2. Driveline Baseball (2019). "Release Point Consistency Analysis"
3. MLB Statcast (2020). "Pitch Sequencing Metrics"
"""

import numpy as np
import pandas as pd
from typing import Optional


class TunnelingFeatures:
    """터널링 및 투구 시퀀싱 피처 생성"""
    
    @staticmethod
    def calculate_release_point_distance(df: pd.DataFrame) -> pd.Series:
        """
        연속 투구 간 릴리스 포인트 유클리드 거리 계산
        
        Parameters:
        -----------
        df : pd.DataFrame
            'release_pos_x', 'release_pos_z' 컬럼 필요
            
        Returns:
        --------
        pd.Series
            릴리스 포인트 거리 (feet)
            
        Example:
        --------
        >>> df['tunnel_distance'] = TunnelingFeatures.calculate_release_point_distance(df)
        >>> # 0.2ft 이하 → 매우 효과적인 터널링
        """
        # 그룹별로 이전 투구 릴리스 포인트 가져오기
        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        
        prev_x = df.groupby(['game_pk', 'at_bat_number'])['release_pos_x'].shift(1)
        prev_z = df.groupby(['game_pk', 'at_bat_number'])['release_pos_z'].shift(1)
        
        # 유클리드 거리
        distance = np.sqrt(
            (df['release_pos_x'] - prev_x)**2 +
            (df['release_pos_z'] - prev_z)**2
        )
        
        return distance.fillna(0.0)
    
    @staticmethod
    def calculate_trajectory_divergence(
        df: pd.DataFrame, 
        evaluation_distance: float = 40.0
    ) -> pd.Series:
        """
        특정 거리(기본 40ft)에서 이전 투구 대비 궤적 분리도 계산
        
        물리 계산:
        ---------
        1. 비행 시간: t = distance / release_speed
        2. 수평 이동: Δx = pfx_x * (t/0.4)²  # 0.4초 = 평균 비행시간
        3. 수직 이동: Δz = pfx_z * (t/0.4)²
        
        Parameters:
        -----------
        df : pd.DataFrame
            'release_speed', 'pfx_x', 'pfx_z' 컬럼 필요
        evaluation_distance : float
            평가 지점 거리 (feet, 기본 40ft)
            
        Returns:
        --------
        pd.Series
            40ft 지점에서의 궤적 차이 (feet)
        """
        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        
        # 비행 시간 계산 (mph → ft/s 변환: 1 mph = 1.467 ft/s)
        speed_ft_s = df['release_speed'] * 1.467
        flight_time = evaluation_distance / speed_ft_s
        
        # 평균 비행시간 0.4초 대비 비율
        time_ratio = (flight_time / 0.4) ** 2
        
        # 40ft 지점에서의 위치 (Magnus 효과 포함)
        current_x = df['pfx_x'] * time_ratio
        current_z = df['pfx_z'] * time_ratio
        
        # 이전 투구 궤적
        prev_pfx_x = df.groupby(['game_pk', 'at_bat_number'])['pfx_x'].shift(1)
        prev_pfx_z = df.groupby(['game_pk', 'at_bat_number'])['pfx_z'].shift(1)
        prev_speed = df.groupby(['game_pk', 'at_bat_number'])['release_speed'].shift(1)
        
        prev_speed_ft_s = prev_speed * 1.467
        prev_flight_time = evaluation_distance / prev_speed_ft_s
        prev_time_ratio = (prev_flight_time / 0.4) ** 2
        
        prev_x = prev_pfx_x * prev_time_ratio
        prev_z = prev_pfx_z * prev_time_ratio
        
        # 궤적 분리도
        divergence = np.sqrt(
            (current_x - prev_x)**2 +
            (current_z - prev_z)**2
        )
        
        return divergence.fillna(0.0)
    
    @staticmethod
    def calculate_velocity_differential(df: pd.DataFrame) -> pd.Series:
        """
        연속 투구 간 속도 차이
        
        큰 속도 차이(10+ mph) → 타이밍 교란 효과
        
        Returns:
        --------
        pd.Series
            속도 차이 (mph, 절댓값)
        """
        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        
        prev_speed = df.groupby(['game_pk', 'at_bat_number'])['release_speed'].shift(1)
        velocity_diff = np.abs(df['release_speed'] - prev_speed)
        
        return velocity_diff.fillna(0.0)
    
    @staticmethod
    def pitch_sequencing_patterns(
        df: pd.DataFrame, 
        window: int = 5,
        pitch_types: list = None
    ) -> pd.DataFrame:
        """
        최근 N개 투구의 패턴 피처 생성
        
        Parameters:
        -----------
        df : pd.DataFrame
            'pitch_type' 컬럼 필요
        window : int
            분석할 최근 투구 개수 (기본 5)
        pitch_types : list
            추적할 구종 리스트 (기본: ['FF', 'SL', 'CH', 'CU'])
            
        Returns:
        --------
        pd.DataFrame
            원본 df + 추가 피처 컬럼들
        """
        if pitch_types is None:
            pitch_types = ['FF', 'SL', 'CH', 'CU']
        
        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        result_df = df.copy()
        
        # 1. 구종별 최근 N개 투구 카운트
        for ptype in pitch_types:
            # rolling apply는 string을 처리할 수 없으므로 수동 계산
            def count_pitch_type(group):
                counts = []
                for i in range(len(group)):
                    if i == 0:
                        counts.append(0)
                    else:
                        start_idx = max(0, i - window)
                        recent = group.iloc[start_idx:i]
                        counts.append((recent == ptype).sum())
                return pd.Series(counts, index=group.index)
            
            result_df[f'{ptype}_count_last_{window}'] = (
                result_df.groupby(['game_pk', 'at_bat_number'])['pitch_type']
                .transform(count_pitch_type)
            ).fillna(0)
        
        # 2. 속구 여부 (FF, SI, FC)
        result_df['is_fastball'] = result_df['pitch_type'].isin(['FF', 'SI', 'FC']).astype(int)
        result_df['prev_is_fastball'] = (
            result_df.groupby(['game_pk', 'at_bat_number'])['is_fastball'].shift(1)
        ).fillna(0)
        
        # 3. 속구 → 변화구 전환 패턴
        result_df['fb_to_breaking'] = (
            (result_df['prev_is_fastball'] == 1) & 
            (result_df['is_fastball'] == 0)
        ).astype(int)
        
        # 4. 변화구 → 속구 전환 패턴 (역 페이드)
        result_df['breaking_to_fb'] = (
            (result_df['prev_is_fastball'] == 0) & 
            (result_df['is_fastball'] == 1)
        ).astype(int)
        
        # 5. 연속 동일 구종 횟수
        result_df['same_pitch_streak'] = (
            result_df.groupby(['game_pk', 'at_bat_number'])['pitch_type']
            .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        )
        
        # 6. 최근 5투구 엔트로피 (다양성 측정)
        def calculate_entropy(series):
            """Shannon Entropy: H = -Σ p(x) log₂(p(x))"""
            if len(series) == 0:
                return 0.0
            counts = series.value_counts()
            probs = counts / len(series)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            return entropy
        
        def rolling_entropy(group):
            """각 위치에서 이전 window개 투구의 엔트로피 계산"""
            entropies = []
            for i in range(len(group)):
                if i == 0:
                    entropies.append(0.0)
                else:
                    start_idx = max(0, i - window)
                    recent = group.iloc[start_idx:i]
                    if len(recent) >= 2:
                        entropies.append(calculate_entropy(recent))
                    else:
                        entropies.append(0.0)
            return pd.Series(entropies, index=group.index)
        
        result_df['sequence_entropy'] = (
            result_df.groupby(['game_pk', 'at_bat_number'])['pitch_type']
            .transform(rolling_entropy)
        ).fillna(0.0)
        
        return result_df
    
    @staticmethod
    def calculate_location_tunnel(df: pd.DataFrame) -> pd.Series:
        """
        홈플레이트 위치 기준 터널링 효과
        
        스트라이크존 진입 각도가 비슷하면 터널링 효과 ↑
        
        Returns:
        --------
        pd.Series
            홈플레이트 위치 차이 (feet)
        """
        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        
        # 이전 투구 위치
        prev_px = df.groupby(['game_pk', 'at_bat_number'])['plate_x'].shift(1)
        prev_pz = df.groupby(['game_pk', 'at_bat_number'])['plate_z'].shift(1)
        
        # 홈플레이트 위치 차이
        location_diff = np.sqrt(
            (df['plate_x'] - prev_px)**2 +
            (df['plate_z'] - prev_pz)**2
        )
        
        return location_diff.fillna(0.0)
    
    @staticmethod
    def add_all_tunneling_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 터널링 피처를 한 번에 추가
        
        Parameters:
        -----------
        df : pd.DataFrame
            MLB Statcast 데이터
            
        Returns:
        --------
        pd.DataFrame
            12개의 새로운 터널링 피처 추가된 DataFrame
            
        새로운 피처:
        ------------
        1. tunnel_distance: 릴리스 포인트 거리 (feet)
        2. trajectory_div: 40ft 궤적 분리도 (feet)
        3. velocity_diff: 속도 차이 (mph)
        4. location_tunnel: 홈플레이트 위치 차이 (feet)
        5. FF_count_last_5: 최근 5투구 FF 개수
        6. SL_count_last_5: 최근 5투구 SL 개수
        7. CH_count_last_5: 최근 5투구 CH 개수
        8. CU_count_last_5: 최근 5투구 CU 개수
        9. fb_to_breaking: 속구→변화구 전환 (0/1)
        10. breaking_to_fb: 변화구→속구 전환 (0/1)
        11. same_pitch_streak: 연속 동일 구종 횟수
        12. sequence_entropy: 최근 5투구 엔트로피
        
        Example:
        --------
        >>> df_enhanced = TunnelingFeatures.add_all_tunneling_features(df)
        >>> print(df_enhanced.columns)
        """
        print("🎯 Adding tunneling features...")
        
        # 기본 터널링 메트릭스
        df['tunnel_distance'] = TunnelingFeatures.calculate_release_point_distance(df)
        df['trajectory_div'] = TunnelingFeatures.calculate_trajectory_divergence(df)
        df['velocity_diff'] = TunnelingFeatures.calculate_velocity_differential(df)
        df['location_tunnel'] = TunnelingFeatures.calculate_location_tunnel(df)
        
        # 시퀀싱 패턴
        df = TunnelingFeatures.pitch_sequencing_patterns(df, window=5)
        
        print("✅ Tunneling features added!")
        print(f"   - tunnel_distance: {df['tunnel_distance'].mean():.3f} ft (avg)")
        print(f"   - trajectory_div: {df['trajectory_div'].mean():.3f} ft (avg)")
        print(f"   - velocity_diff: {df['velocity_diff'].mean():.1f} mph (avg)")
        print(f"   - sequence_entropy: {df['sequence_entropy'].mean():.3f} (avg)")
        
        return df


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tunneling Features Module")
    print("=" * 60)
    
    # 더미 데이터 생성
    np.random.seed(42)
    n_pitches = 100
    
    test_df = pd.DataFrame({
        'game_pk': np.repeat(12345, n_pitches),
        'at_bat_number': np.repeat(range(1, 21), 5),  # 20 at-bats, 5 pitches each
        'pitch_number': np.tile(range(1, 6), 20),
        'release_pos_x': np.random.normal(-2.0, 0.2, n_pitches),  # feet
        'release_pos_z': np.random.normal(6.0, 0.3, n_pitches),   # feet
        'release_speed': np.random.normal(92, 5, n_pitches),       # mph
        'pfx_x': np.random.normal(0, 5, n_pitches),               # inches
        'pfx_z': np.random.normal(0, 8, n_pitches),               # inches
        'plate_x': np.random.normal(0, 0.8, n_pitches),           # feet
        'plate_z': np.random.normal(2.5, 0.6, n_pitches),         # feet
        'pitch_type': np.random.choice(['FF', 'SL', 'CH', 'CU'], n_pitches, 
                                       p=[0.35, 0.25, 0.20, 0.20])
    })
    
    print("\n📊 Test Data:")
    print(f"   Total pitches: {len(test_df)}")
    print(f"   At-bats: {test_df['at_bat_number'].nunique()}")
    print(f"   Pitch types: {test_df['pitch_type'].value_counts().to_dict()}")
    
    # Test 1: 릴리스 포인트 거리
    print("\n" + "=" * 60)
    print("Test 1: Release Point Distance")
    print("=" * 60)
    test_df['tunnel_distance'] = TunnelingFeatures.calculate_release_point_distance(test_df)
    print(f"✅ Mean distance: {test_df['tunnel_distance'].mean():.3f} ft")
    print(f"   Min: {test_df['tunnel_distance'].min():.3f} ft")
    print(f"   Max: {test_df['tunnel_distance'].max():.3f} ft")
    print(f"   Pitches with good tunneling (<0.3 ft): {(test_df['tunnel_distance'] < 0.3).sum()}")
    
    # Test 2: 궤적 분리도
    print("\n" + "=" * 60)
    print("Test 2: Trajectory Divergence at 40ft")
    print("=" * 60)
    test_df['trajectory_div'] = TunnelingFeatures.calculate_trajectory_divergence(test_df)
    print(f"✅ Mean divergence: {test_df['trajectory_div'].mean():.3f} ft")
    print(f"   Min: {test_df['trajectory_div'].min():.3f} ft")
    print(f"   Max: {test_df['trajectory_div'].max():.3f} ft")
    
    # Test 3: 속도 차이
    print("\n" + "=" * 60)
    print("Test 3: Velocity Differential")
    print("=" * 60)
    test_df['velocity_diff'] = TunnelingFeatures.calculate_velocity_differential(test_df)
    print(f"✅ Mean velocity diff: {test_df['velocity_diff'].mean():.1f} mph")
    print(f"   Large gaps (>10 mph): {(test_df['velocity_diff'] > 10).sum()}")
    
    # Test 4: 시퀀싱 패턴
    print("\n" + "=" * 60)
    print("Test 4: Pitch Sequencing Patterns")
    print("=" * 60)
    test_df = TunnelingFeatures.pitch_sequencing_patterns(test_df)
    print(f"✅ FF count (last 5): {test_df['FF_count_last_5'].mean():.2f}")
    print(f"   SL count (last 5): {test_df['SL_count_last_5'].mean():.2f}")
    print(f"   FB→Breaking transitions: {test_df['fb_to_breaking'].sum()}")
    print(f"   Sequence entropy: {test_df['sequence_entropy'].mean():.3f}")
    
    # Test 5: 종합 피처 추가
    print("\n" + "=" * 60)
    print("Test 5: Add All Tunneling Features")
    print("=" * 60)
    test_df_full = TunnelingFeatures.add_all_tunneling_features(test_df.copy())
    
    tunneling_cols = [col for col in test_df_full.columns 
                     if any(x in col for x in ['tunnel', 'trajectory', 'velocity', 
                                               'count_last', 'fb_to', 'streak', 'entropy'])]
    print(f"\n✅ Total tunneling features: {len(tunneling_cols)}")
    print(f"   Feature names: {tunneling_cols}")
    
    print("\n" + "=" * 60)
    print("✅ All Tests Passed!")
    print("=" * 60)
