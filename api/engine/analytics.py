import pandas as pd
import numpy as np

class VolumetricAnalytics:
    """
    [v7.0 Phase 3] Volumetric Analysis Engine
    타자의 약점을 2D(Plate X,Z)가 아닌 3D(Velocity, HorzBreak, VertBreak) 물리 공간에서 분석합니다.
    """
    def generate_volumetric_hotzone(self, batter_df: pd.DataFrame):
        """
        타자가 상대한 투구들을 3D 물리 공간에 매핑하고, 결과(헛스윙/타격)를 색상으로 구분합니다.
        """
        if batter_df.empty:
            return []

        # 분석에 필요한 컬럼만 추출 & 결측치 제거
        required_cols = ['release_speed', 'pfx_x', 'pfx_z', 'description', 'pitch_type']
        df = batter_df[required_cols].dropna().copy()
        
        # 데이터가 너무 많으면 최근 500구만 사용 (성능 최적화)
        if len(df) > 500:
            df = df.iloc[-500:]

        # 결과 카테고리화 (Color Coding)
        # Red: 헛스윙/스트라이크 (Weakness)
        # Blue: 타격/인플레이 (Strength)
        # Gray: 볼 (Take)
        def categorize_result(desc):
            if desc in ['swinging_strike', 'swinging_strike_blocked', 'called_strike', 'foul_tip']:
                return 'Weakness (Strike/Whiff)'
            elif desc in ['hit_into_play', 'foul']:
                return 'Strength (Contact)'
            else:
                return 'Neutral (Ball)'

        df['result_category'] = df['description'].apply(categorize_result)
        
        # 시각화용 데이터 리스트 변환
        volumetric_data = []
        for _, row in df.iterrows():
            volumetric_data.append({
                "velocity": row['release_speed'],
                "horz_break": row['pfx_x'],
                "vert_break": row['pfx_z'],
                "result": row['result_category'],
                "pitch_type": row['pitch_type']
            })
            
        return volumetric_data