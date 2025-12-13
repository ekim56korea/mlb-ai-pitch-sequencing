import pandas as pd
import numpy as np
from scipy import stats

class DataPreprocessor:
    """
    [v5.1] Data Pipeline: Cleaning + Feature Engineering
    """
    
    def __init__(self):
        self.CRITICAL_COLS = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z']

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 clean_data 로직 유지"""
        if df is None or df.empty: return pd.DataFrame()
        
        # 1. 결측치 제거
        df_clean = df.dropna(subset=self.CRITICAL_COLS).copy()
        
        # 2. 이상치 제거 (Z-Score)
        if len(df_clean) > 30:
            z_scores = np.abs(stats.zscore(df_clean[self.CRITICAL_COLS]))
            df_clean = df_clean[(z_scores < 3).all(axis=1)]
            
        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Feature Engineering] 도메인 지식을 활용한 파생 변수 생성
        """
        df_eng = df.copy()
        
        # 컬럼 이름 통일 (입력 데이터가 extension으로 들어올 경우 release_extension으로 변경)
        if 'extension' in df_eng.columns and 'release_extension' not in df_eng.columns:
            df_eng.rename(columns={'extension': 'release_extension'}, inplace=True)
            
        # 기본값이 없는 경우 평균값 등으로 채움 (안전장치)
        if 'release_extension' not in df_eng.columns:
            df_eng['release_extension'] = 6.0

        # 1. Effective Velocity (체감 구속)
        # 공식: 실구속 + (익스텐션 - 평균6.0) * 1.2
        # 익스텐션이 길수록 타자 체감 구속이 빨라짐
        df_eng['effective_velo'] = df_eng['release_speed'] + (df_eng['release_extension'] - 6.0) * 1.2
        
        # 2. Velo-Movement Interaction (구속과 수직 무브먼트의 시너지)
        # 빠른 공이 덜 떨어질수록(High PFX_Z) 위력이 배가됨 (상호작용 항)
        df_eng['velo_pfx_z_interaction'] = df_eng['release_speed'] * df_eng['pfx_z']
        
        # 3. Spin Efficiency Proxy (회전 효율 근사치)
        # 회전수 대비 무브먼트가 얼마나 큰가?
        total_movement = np.sqrt(df_eng['pfx_x']**2 + df_eng['pfx_z']**2)
        df_eng['movement_per_spin'] = total_movement / (df_eng['release_spin_rate'] + 1e-9)
        
        return df_eng

    def apply_bayesian_shrinkage(self, raw_score, sample_size, league_avg=100, confidence_constant=50):
        """기존 메서드 유지"""
        if sample_size == 0: return league_avg, "None"
        weight = sample_size / (sample_size + confidence_constant)
        adjusted_score = (weight * raw_score) + ((1 - weight) * league_avg)
        
        if sample_size < 20: confidence = "Low"
        elif sample_size < 100: confidence = "Medium"
        else: confidence = "High"
        
        return round(adjusted_score, 1), confidence