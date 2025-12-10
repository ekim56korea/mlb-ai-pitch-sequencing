import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

class BatterClustering:
    """
    SRS REQ-AI-02: 타자의 스윙/테이크 성향을 기반으로 5개 그룹으로 클러스터링합니다.
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.feature_columns = ['swing_rate', 'whiff_rate', 'chase_rate']

    def preprocess_data(self, df: pd.DataFrame):
        """
        Raw Statcast 데이터를 타자별 요약 통계(Feature)로 변환합니다.
        """
        # 데이터 복사본 생성 (원본 보호)
        data = df.copy()

        # [🚨 긴급 수정] 구종 정보가 없는(NaN) 쓰레기 데이터 제거
        # 이를 처리하지 않으면 스윙률이 1.0을 넘는 버그가 발생함
        data = data.dropna(subset=['pitch_type'])

        # 1. 스윙 여부 정의
        swing_events = [
            'hit_into_play', 'foul', 'swinging_strike', 
            'swinging_strike_blocked', 'foul_tip'
        ]
        data['is_swing'] = data['description'].isin(swing_events)

        # 2. 헛스윙(Whiff) 여부
        whiff_events = ['swinging_strike', 'swinging_strike_blocked']
        data['is_whiff'] = data['description'].isin(whiff_events)

        # 3. 유인구(Chase) 여부: 존(zone)이 1~9가 아니면(11~14) 볼(Ball)로 간주
        data['is_out_of_zone'] = data['zone'] > 9
        
        # [수정됨] 4. Chase(유인구 스윙) 여부를 미리 계산 (에러 방지)
        # 조건: 존 바깥 공(is_out_of_zone) AND 스윙함(is_swing)
        data['is_chase'] = data['is_out_of_zone'] & data['is_swing']

        # --- 타자별 그룹화 및 비율 계산 ---
        # 수정됨: lambda 함수를 제거하고 미리 계산된 컬럼을 합산(sum)합니다.
        batter_stats = data.groupby('batter').agg(
            total_pitches=('pitch_type', 'count'),
            total_swings=('is_swing', 'sum'),
            total_whiffs=('is_whiff', 'sum'),
            out_of_zone_pitches=('is_out_of_zone', 'sum'),
            chases=('is_chase', 'sum') 
        ).reset_index()

        # 최소 5구 이상 상대한 타자만 분석 (데이터 노이즈 제거)
        batter_stats = batter_stats[batter_stats['total_pitches'] >= 5]

        # 비율(Rate) 계산
        batter_stats['swing_rate'] = batter_stats['total_swings'] / batter_stats['total_pitches']
        
        # 헛스윙률 (스윙 대비 헛스윙)
        batter_stats['whiff_rate'] = 0.0
        mask_swing = batter_stats['total_swings'] > 0
        batter_stats.loc[mask_swing, 'whiff_rate'] = (
            batter_stats.loc[mask_swing, 'total_whiffs'] / batter_stats.loc[mask_swing, 'total_swings']
        )

        # 추격률(Chase Rate): 존 바깥 공에 스윙한 비율
        batter_stats['chase_rate'] = 0.0
        mask_ooz = batter_stats['out_of_zone_pitches'] > 0
        batter_stats.loc[mask_ooz, 'chase_rate'] = (
            batter_stats.loc[mask_ooz, 'chases'] / batter_stats.loc[mask_ooz, 'out_of_zone_pitches']
        )

        # 필요한 컬럼만 리턴
        return batter_stats[['batter'] + self.feature_columns].set_index('batter')

    def train(self, df: pd.DataFrame):
        """
        데이터를 받아 모델을 학습시킵니다.
        """
        print("📊 타자 데이터 전처리 중...")
        features = self.preprocess_data(df)
        
        print(f"🤖 {len(features)}명의 타자를 대상으로 학습 시작...")
        # 데이터 정규화 (스케일링)
        scaled_features = self.scaler.fit_transform(features)
        
        # K-Means 학습
        self.model.fit(scaled_features)
        
        features['cluster'] = self.model.labels_
        return features

    def save_model(self, path='api/engine/batter_cluster_model.pkl'):
        """학습된 모델 저장"""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"💾 모델이 저장되었습니다: {path}")