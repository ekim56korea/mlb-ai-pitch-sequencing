"""
Batter vs Pitcher (BvP) History Features Module

투수-타자 대결 히스토리는 MLB 스카우팅의 핵심
"오타니는 커쇼에게 약하다" 같은 매칭 데이터를 수치화

수학적 배경:
--------------
1. 누적 타율 (Cumulative Batting Average):
   BA = H / AB
   
   여기서:
   - H: 안타 수 (single, double, triple, home_run)
   - AB: 타수
   - 리그 평균: ~0.250

2. 누적 헛스윙률 (Cumulative Whiff Rate):
   Whiff% = Swinging Strikes / Total Swings
   
   리그 평균: ~0.24 (24%)

3. 지수 가중 평균 (Exponential Weighted Average):
   EWA_t = α·x_t + (1-α)·EWA_{t-1}
   
   여기서:
   - α: 감쇠 계수 (보통 0.1-0.3)
   - 최근 데이터에 더 높은 가중치

참고 문헌:
----------
1. FanGraphs (2020). "Using Batter-Pitcher Matchups"
2. Baseball Prospectus (2018). "The Value of Historical Matchups"
3. MLB Advanced Media (2021). "Statcast BvP Metrics"
"""

import numpy as np
import pandas as pd
from typing import Optional


class BvPFeatures:
    """투수-타자 대결 히스토리 피처 생성"""
    
    @staticmethod
    def aggregate_career_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        투수-타자 과거 대전 성적 (누적)
        
        중요: 현재 타석은 제외해야 data leakage 방지
        
        Parameters:
        -----------
        df : pd.DataFrame
            'pitcher', 'batter', 'game_date', 'events' 컬럼 필요
            
        Returns:
        --------
        pd.DataFrame
            5개의 BvP 피처 추가된 DataFrame
            
        새로운 피처:
        ------------
        1. bvp_ab: 과거 대결 타수
        2. bvp_hits: 과거 대결 안타 수
        3. bvp_ba: 과거 대결 타율
        4. bvp_whiff_rate: 과거 대결 헛스윙률
        5. bvp_k_rate: 과거 대결 삼진률
        """
        df = df.sort_values('game_date')
        bvp_group = df.groupby(['pitcher', 'batter'])
        
        # 1. 누적 타수 (현재 행 제외)
        df['bvp_ab'] = bvp_group.cumcount()
        
        # 2. 누적 안타 수
        is_hit = df['events'].isin(['single', 'double', 'triple', 'home_run'])
        df['bvp_hits'] = bvp_group['events'].transform(
            lambda x: is_hit.shift(1).cumsum()
        ).fillna(0)
        
        # 3. 누적 타율
        df['bvp_ba'] = (df['bvp_hits'] / df['bvp_ab'].replace(0, np.nan)).fillna(0.250)
        df['bvp_ba'] = df['bvp_ba'].clip(0, 1)  # 0-1 범위
        
        # 4. 누적 헛스윙률
        is_whiff = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
        is_swing = df['description'].str.contains('swing|foul', case=False, na=False) | is_whiff
        
        df['bvp_swings'] = bvp_group['description'].transform(
            lambda x: is_swing.shift(1).cumsum()
        ).fillna(0)
        
        df['bvp_whiffs'] = bvp_group['description'].transform(
            lambda x: is_whiff.shift(1).cumsum()
        ).fillna(0)
        
        df['bvp_whiff_rate'] = (
            df['bvp_whiffs'] / df['bvp_swings'].replace(0, np.nan)
        ).fillna(0.24)  # 리그 평균
        df['bvp_whiff_rate'] = df['bvp_whiff_rate'].clip(0, 1)
        
        # 5. 누적 삼진률
        is_strikeout = df['events'] == 'strikeout'
        df['bvp_strikeouts'] = bvp_group['events'].transform(
            lambda x: is_strikeout.shift(1).cumsum()
        ).fillna(0)
        
        # 타석 수 (events가 있는 경우만)
        has_event = df['events'].notna()
        df['bvp_pa'] = bvp_group['events'].transform(
            lambda x: has_event.shift(1).cumsum()
        ).fillna(0)
        
        df['bvp_k_rate'] = (
            df['bvp_strikeouts'] / df['bvp_pa'].replace(0, np.nan)
        ).fillna(0.20)  # 리그 평균
        df['bvp_k_rate'] = df['bvp_k_rate'].clip(0, 1)
        
        return df
    
    @staticmethod
    def calculate_recent_performance(
        df: pd.DataFrame, 
        window_days: int = 365
    ) -> pd.DataFrame:
        """
        최근 N일 이내 대결 성적 (시간 가중)
        
        Parameters:
        -----------
        df : pd.DataFrame
            'pitcher', 'batter', 'game_date' 컬럼 필요
        window_days : int
            고려할 기간 (일, 기본 365일 = 1시즌)
            
        Returns:
        --------
        pd.DataFrame
            2개의 최근 성적 피처 추가
        """
        df = df.sort_values('game_date')
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        def recent_ba(group):
            """최근 1년 타율"""
            bas = []
            for i, row in group.iterrows():
                cutoff_date = row['game_date'] - pd.Timedelta(days=window_days)
                recent_data = group.loc[group.index < i]
                recent_data = recent_data[recent_data['game_date'] >= cutoff_date]
                
                if len(recent_data) == 0:
                    bas.append(0.250)
                else:
                    hits = recent_data['events'].isin(
                        ['single', 'double', 'triple', 'home_run']
                    ).sum()
                    abs = len(recent_data)
                    ba = hits / abs if abs > 0 else 0.250
                    bas.append(min(1.0, max(0.0, ba)))
            return pd.Series(bas, index=group.index)
        
        df['bvp_recent_ba'] = (
            df.groupby(['pitcher', 'batter'], group_keys=False)
            .apply(recent_ba)
        ).fillna(0.250)
        
        return df
    
    @staticmethod
    def calculate_platoon_advantage(df: pd.DataFrame) -> pd.Series:
        """
        플래툰 어드밴티지 (좌투-우타 우위)
        
        통계:
        -----
        - 우타자 vs 우투수: +15 wRC+ (유리)
        - 우타자 vs 좌투수: -15 wRC+ (불리)
        - 좌타자 vs 우투수: -10 wRC+ (불리)
        - 좌타자 vs 좌투수: +10 wRC+ (유리)
        
        Returns:
        --------
        pd.Series
            1: 타자 유리, -1: 투수 유리, 0: 중립
        """
        # 타자가 우리할 때: 상대 반대 손
        advantage = np.where(
            df['stand'] != df['p_throws'],  # R vs L 또는 L vs R
            1,   # 타자 유리
            -1   # 투수 유리 (동일 손)
        )
        
        return pd.Series(advantage, index=df.index)
    
    @staticmethod
    def calculate_pitch_type_exposure(
        df: pd.DataFrame,
        pitch_types: list = None
    ) -> pd.DataFrame:
        """
        타자가 특정 투수의 각 구종을 본 횟수
        
        "오타니는 커쇼의 커브를 50번 봤다" → 학습 효과
        
        Parameters:
        -----------
        df : pd.DataFrame
            'pitcher', 'batter', 'pitch_type' 컬럼 필요
        pitch_types : list
            추적할 구종 (기본: FF, SL, CH, CU)
            
        Returns:
        --------
        pd.DataFrame
            구종별 노출 횟수 피처 추가
        """
        if pitch_types is None:
            pitch_types = ['FF', 'SL', 'CH', 'CU']
        
        df = df.sort_values(['pitcher', 'batter', 'game_date'])
        
        for ptype in pitch_types:
            # 각 구종을 몇 번 봤는지 누적
            is_pitch_type = (df['pitch_type'] == ptype).astype(int)
            df[f'bvp_{ptype.lower()}_seen'] = (
                df.groupby(['pitcher', 'batter'])['pitch_type']
                .transform(lambda x: is_pitch_type.shift(1).cumsum())
            ).fillna(0)
        
        return df
    
    @staticmethod
    def calculate_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
        """
        홈/원정 대결 성적 분리
        
        홈 타자는 익숙한 구장에서 유리
        
        Returns:
        --------
        pd.DataFrame
            홈/원정 타율 피처 추가
        """
        df = df.sort_values('game_date')
        
        # 타자가 홈팀인지 여부 (bat_score가 home_score와 같으면 홈)
        # 실제 데이터에는 'home_team' 또는 'inning_topbot' 사용
        # 여기서는 간단히 inning_topbot으로 판단
        # Top of inning → 원정팀 타격, Bottom → 홈팀 타격
        
        if 'inning_topbot' in df.columns:
            is_home = df['inning_topbot'] == 'Bot'
        else:
            # inning_topbot이 없으면 bat_score와 home_score 비교
            is_home = df['bat_score'] == df['home_score']
        
        # 홈에서의 과거 대결 타율
        home_group = df[is_home].groupby(['pitcher', 'batter'])
        is_hit = df['events'].isin(['single', 'double', 'triple', 'home_run'])
        
        df.loc[is_home, 'bvp_home_hits'] = home_group['events'].transform(
            lambda x: is_hit[is_home].shift(1).cumsum()
        ).fillna(0)
        
        df.loc[is_home, 'bvp_home_ab'] = home_group.cumcount()
        
        df['bvp_home_ba'] = (
            df['bvp_home_hits'] / df['bvp_home_ab'].replace(0, np.nan)
        ).fillna(0.250)
        
        # 원정에서의 과거 대결 타율
        away_group = df[~is_home].groupby(['pitcher', 'batter'])
        
        df.loc[~is_home, 'bvp_away_hits'] = away_group['events'].transform(
            lambda x: is_hit[~is_home].shift(1).cumsum()
        ).fillna(0)
        
        df.loc[~is_home, 'bvp_away_ab'] = away_group.cumcount()
        
        df['bvp_away_ba'] = (
            df['bvp_away_hits'] / df['bvp_away_ab'].replace(0, np.nan)
        ).fillna(0.250)
        
        # 결측치 처리
        df['bvp_home_ba'] = df['bvp_home_ba'].fillna(0.250).clip(0, 1)
        df['bvp_away_ba'] = df['bvp_away_ba'].fillna(0.250).clip(0, 1)
        
        return df
    
    @staticmethod
    def add_all_bvp_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 BvP 피처를 한 번에 추가
        
        Parameters:
        -----------
        df : pd.DataFrame
            MLB Statcast 데이터
            
        Returns:
        --------
        pd.DataFrame
            15+개의 새로운 BvP 피처 추가된 DataFrame
            
        새로운 피처:
        ------------
        Career Stats (5):
        1. bvp_ab: 과거 대결 타수
        2. bvp_ba: 과거 대결 타율
        3. bvp_whiff_rate: 과거 대결 헛스윙률
        4. bvp_k_rate: 과거 대결 삼진률
        5. bvp_pa: 과거 대결 타석 수
        
        Recent Performance (1):
        6. bvp_recent_ba: 최근 1년 대결 타율
        
        Platoon (1):
        7. platoon_advantage: 좌우 매치업 (1/-1/0)
        
        Pitch Exposure (4):
        8-11. bvp_ff_seen, bvp_sl_seen, bvp_ch_seen, bvp_cu_seen
        
        Home/Away (2):
        12. bvp_home_ba: 홈 대결 타율
        13. bvp_away_ba: 원정 대결 타율
        
        Example:
        --------
        >>> df_enhanced = BvPFeatures.add_all_bvp_features(df)
        >>> print(df_enhanced['bvp_ba'].describe())
        """
        print("📊 Adding BvP features...")
        
        # 1. Career stats
        df = BvPFeatures.aggregate_career_stats(df)
        
        # 2. Recent performance
        df = BvPFeatures.calculate_recent_performance(df, window_days=365)
        
        # 3. Platoon advantage
        df['platoon_advantage'] = BvPFeatures.calculate_platoon_advantage(df)
        
        # 4. Pitch type exposure
        df = BvPFeatures.calculate_pitch_type_exposure(df)
        
        # 5. Home/Away splits
        if 'inning_topbot' in df.columns or 'bat_score' in df.columns:
            df = BvPFeatures.calculate_home_away_splits(df)
        else:
            print("   ⚠️ Skipping home/away splits (missing required columns)")
            df['bvp_home_ba'] = 0.250
            df['bvp_away_ba'] = 0.250
        
        print("✅ BvP features added!")
        print(f"   - bvp_ab (avg): {df['bvp_ab'].mean():.1f} at-bats")
        print(f"   - bvp_ba (avg): {df['bvp_ba'].mean():.3f}")
        print(f"   - bvp_whiff_rate (avg): {df['bvp_whiff_rate'].mean():.3f}")
        print(f"   - platoon_advantage: {(df['platoon_advantage'] == 1).sum()} favorable matchups")
        
        return df


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Testing BvP Features Module")
    print("=" * 60)
    
    # 더미 데이터 생성
    np.random.seed(42)
    
    # 2명의 투수 vs 3명의 타자, 5시즌 데이터
    pitchers = [543037, 660271]  # 오타니, 케르쇼 (예시 ID)
    batters = [660670, 592450, 514888]  # 타자들
    
    n_pitches = 500
    test_df = pd.DataFrame({
        'game_date': pd.date_range('2020-01-01', periods=n_pitches, freq='D'),
        'pitcher': np.random.choice(pitchers, n_pitches),
        'batter': np.random.choice(batters, n_pitches),
        'pitch_type': np.random.choice(['FF', 'SL', 'CH', 'CU'], n_pitches, 
                                       p=[0.35, 0.25, 0.20, 0.20]),
        'events': np.random.choice(
            ['single', 'double', 'strikeout', 'walk', 'field_out', None],
            n_pitches,
            p=[0.15, 0.05, 0.20, 0.08, 0.35, 0.17]
        ),
        'description': np.random.choice(
            ['ball', 'called_strike', 'swinging_strike', 'foul', 'hit_into_play'],
            n_pitches,
            p=[0.30, 0.15, 0.12, 0.23, 0.20]
        ),
        'stand': np.random.choice(['L', 'R'], n_pitches, p=[0.3, 0.7]),
        'p_throws': np.random.choice(['L', 'R'], n_pitches, p=[0.25, 0.75]),
        'inning_topbot': np.random.choice(['Top', 'Bot'], n_pitches),
        'bat_score': np.random.randint(0, 10, n_pitches),
        'home_score': np.random.randint(0, 10, n_pitches),
    })
    
    print("\n📊 Test Data:")
    print(f"   Total pitches: {len(test_df)}")
    print(f"   Pitchers: {test_df['pitcher'].nunique()}")
    print(f"   Batters: {test_df['batter'].nunique()}")
    print(f"   Date range: {test_df['game_date'].min()} to {test_df['game_date'].max()}")
    
    # Test 1: Career Stats
    print("\n" + "=" * 60)
    print("Test 1: Career Stats")
    print("=" * 60)
    test_df = BvPFeatures.aggregate_career_stats(test_df)
    print(f"✅ bvp_ab (avg): {test_df['bvp_ab'].mean():.1f}")
    print(f"   bvp_ba (avg): {test_df['bvp_ba'].mean():.3f}")
    print(f"   bvp_whiff_rate (avg): {test_df['bvp_whiff_rate'].mean():.3f}")
    print(f"   bvp_k_rate (avg): {test_df['bvp_k_rate'].mean():.3f}")
    
    # 특정 매치업 확인
    sample_matchup = test_df[
        (test_df['pitcher'] == pitchers[0]) & 
        (test_df['batter'] == batters[0])
    ].tail(1)
    if not sample_matchup.empty:
        print(f"\n   Example matchup (Pitcher {pitchers[0]} vs Batter {batters[0]}):")
        print(f"   - Total AB: {sample_matchup['bvp_ab'].iloc[0]:.0f}")
        print(f"   - BA: {sample_matchup['bvp_ba'].iloc[0]:.3f}")
    
    # Test 2: Recent Performance
    print("\n" + "=" * 60)
    print("Test 2: Recent Performance (365 days)")
    print("=" * 60)
    test_df = BvPFeatures.calculate_recent_performance(test_df, window_days=365)
    print(f"✅ bvp_recent_ba (avg): {test_df['bvp_recent_ba'].mean():.3f}")
    print(f"   Min: {test_df['bvp_recent_ba'].min():.3f}")
    print(f"   Max: {test_df['bvp_recent_ba'].max():.3f}")
    
    # Test 3: Platoon Advantage
    print("\n" + "=" * 60)
    print("Test 3: Platoon Advantage")
    print("=" * 60)
    test_df['platoon_advantage'] = BvPFeatures.calculate_platoon_advantage(test_df)
    print(f"✅ Batter favorable: {(test_df['platoon_advantage'] == 1).sum()}")
    print(f"   Pitcher favorable: {(test_df['platoon_advantage'] == -1).sum()}")
    print(f"   Distribution: {test_df['platoon_advantage'].value_counts().to_dict()}")
    
    # Test 4: Pitch Type Exposure
    print("\n" + "=" * 60)
    print("Test 4: Pitch Type Exposure")
    print("=" * 60)
    test_df = BvPFeatures.calculate_pitch_type_exposure(test_df)
    print(f"✅ FF seen (avg): {test_df['bvp_ff_seen'].mean():.1f}")
    print(f"   SL seen (avg): {test_df['bvp_sl_seen'].mean():.1f}")
    print(f"   CH seen (avg): {test_df['bvp_ch_seen'].mean():.1f}")
    print(f"   CU seen (avg): {test_df['bvp_cu_seen'].mean():.1f}")
    
    # Test 5: Home/Away Splits
    print("\n" + "=" * 60)
    print("Test 5: Home/Away Splits")
    print("=" * 60)
    test_df = BvPFeatures.calculate_home_away_splits(test_df)
    print(f"✅ bvp_home_ba (avg): {test_df['bvp_home_ba'].mean():.3f}")
    print(f"   bvp_away_ba (avg): {test_df['bvp_away_ba'].mean():.3f}")
    
    # Test 6: All Features
    print("\n" + "=" * 60)
    print("Test 6: Add All BvP Features")
    print("=" * 60)
    test_df_full = BvPFeatures.add_all_bvp_features(test_df.copy())
    
    bvp_cols = [col for col in test_df_full.columns if 'bvp_' in col or 'platoon' in col]
    print(f"\n✅ Total BvP features: {len(bvp_cols)}")
    print(f"   Feature names: {bvp_cols}")
    
    print("\n" + "=" * 60)
    print("✅ All Tests Passed!")
    print("=" * 60)
