import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
# ==========================================
# 1. 타자 유형 분류 모델 (Existing)
# ==========================================
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
        data = df.copy()
        data = data.dropna(subset=['pitch_type'])

        swing_events = ['hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip']
        data['is_swing'] = data['description'].isin(swing_events)

        whiff_events = ['swinging_strike', 'swinging_strike_blocked']
        data['is_whiff'] = data['description'].isin(whiff_events)

        data['is_out_of_zone'] = data['zone'] > 9
        data['is_chase'] = data['is_out_of_zone'] & data['is_swing']

        batter_stats = data.groupby('batter').agg(
            total_pitches=('pitch_type', 'count'),
            total_swings=('is_swing', 'sum'),
            total_whiffs=('is_whiff', 'sum'),
            out_of_zone_pitches=('is_out_of_zone', 'sum'),
            chases=('is_chase', 'sum') 
        ).reset_index()

        batter_stats = batter_stats[batter_stats['total_pitches'] >= 5]

        batter_stats['swing_rate'] = batter_stats['total_swings'] / batter_stats['total_pitches']
        
        batter_stats['whiff_rate'] = 0.0
        mask_swing = batter_stats['total_swings'] > 0
        batter_stats.loc[mask_swing, 'whiff_rate'] = (
            batter_stats.loc[mask_swing, 'total_whiffs'] / batter_stats.loc[mask_swing, 'total_swings']
        )

        batter_stats['chase_rate'] = 0.0
        mask_ooz = batter_stats['out_of_zone_pitches'] > 0
        batter_stats.loc[mask_ooz, 'chase_rate'] = (
            batter_stats.loc[mask_ooz, 'chases'] / batter_stats.loc[mask_ooz, 'out_of_zone_pitches']
        )

        return batter_stats[['batter'] + self.feature_columns].set_index('batter')

    def train(self, df: pd.DataFrame):
        print("📊 타자 데이터 전처리 중...")
        features = self.preprocess_data(df)
        
        print(f"🤖 {len(features)}명의 타자를 대상으로 학습 시작...")
        scaled_features = self.scaler.fit_transform(features)
        
        self.model.fit(scaled_features)
        
        features['cluster'] = self.model.labels_
        return features

    def save_model(self, path='api/engine/batter_cluster_model.pkl'):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"💾 모델이 저장되었습니다: {path}")


# ==========================================
# 2. 타자 노림수 예측 모델 (New Phase 2)
# ==========================================
class GuessHittingModel:
    """
    [v7.0 Phase 2] Bayesian Guess Hitting Model
    타자가 현재 카운트에서 특정 구종을 노리고 있을 확률(Guess Probability)을 추론합니다.
    Logic: P(Guess | Count) ~ Prior(Count) * Likelihood(Batter History)
    """
    def __init__(self):
        # 카운트별 타자들의 일반적인 노림수 사전확률 (MLB Average Priors)
        self.priors = {
            "0-0": {"FF": 0.60, "SL": 0.20, "CH": 0.10, "CB": 0.10}, # 초구 직구 노림
            "3-0": {"FF": 0.95, "SL": 0.05, "CH": 0.00, "CB": 0.00}, # 무조건 직구
            "3-1": {"FF": 0.85, "SL": 0.10, "CH": 0.05, "CB": 0.00}, # 히팅 카운트
            "0-2": {"FF": 0.30, "SL": 0.40, "CH": 0.20, "CB": 0.10}, # 유인구 대비(변화구 예상)
            "Default": {"FF": 0.50, "SL": 0.25, "CH": 0.15, "CB": 0.10}
        }

    def predict_guess_probabilities(self, context, arsenal):
        """
        현재 상황(Context)에서 타자의 구종별 예측 확률 반환
        """
        balls = context.get('balls', 0)
        strikes = context.get('strikes', 0)
        count_key = f"{balls}-{strikes}"
        
        # 1. 사전 확률 가져오기 (없으면 Default)
        probs = self.priors.get(count_key, self.priors["Default"]).copy()
        
        # 2. 투수의 구종(Arsenal)에 맞게 정규화
        # 투수가 던질 수 없는 구종은 확률 0 처리하고 나머지를 재분배
        total_prob = 0
        valid_probs = {}
        
        for pitch in arsenal:
            # 해당 구종에 대한 Prior가 없으면 기본값 0.1 부여
            p = probs.get(pitch, 0.1) 
            valid_probs[pitch] = p
            total_prob += p
            
        # 정규화 (확률 합 = 100%)
        if total_prob > 0:
            for pitch in valid_probs:
                valid_probs[pitch] = round((valid_probs[pitch] / total_prob) * 100, 1)
        
        return valid_probs

    def calculate_risk_penalty(self, pitch_type, guess_probs):
        """
        타자가 노리고 있는 공을 던졌을 때의 위험도(Penalty) 계산
        """
        prob = guess_probs.get(pitch_type, 0)
        
        # 타자가 60% 이상 확신하고 노리는 공이라면 페널티 부여
        if prob > 60:
            return -20.0 # High Danger (홈런 위험)
        elif prob > 40:
            return -5.0  # Moderate Danger
        else:
            return 5.0   # Reverse Guess Bonus (역으로 찌르기 성공)
        
        
        
class SwingTakeModel:
    """
    [v7.0 Phase 3] Lightweight Swing/Take Predictor (CPU-based)
    - GPU 없이 Random Forest나 로직 기반으로 타자의 스윙 확률 예측
    - Cold Start: 데이터가 없으면 물리적 존(Zone) 기반 휴리스틱 사용
    """
    def __init__(self):
        self.model = None
        # 간단한 사전 학습된 가중치가 있다고 가정하거나, 
        # 존 중심에서의 거리에 따른 스윙 확률 분포를 수식화하여 사용 (Zero-Cost Trick)
        
    def predict_swing_prob(self, pitch_type, plate_x, plate_z, count_context):
        """
        특정 투구 위치와 상황에서 타자가 스윙할 확률 반환 (0.0 ~ 1.0)
        """
        # 1. 존 중심에서의 거리 계산
        # 스트라이크 존 중심 (0, 2.5)
        dist = np.sqrt(plate_x**2 + (plate_z - 2.5)**2)
        
        # 2. 기본 스윙 확률 (거리 기반 로지스틱 함수 근사)
        # 중심에 가까울수록 스윙 확률 높음 (약 1.5ft 벗어나면 급격히 하락)
        base_prob = 1.0 / (1.0 + np.exp(4.0 * (dist - 1.2)))
        
        # 3. 카운트 보정 (베이지안 아이디어)
        # 불리한 카운트(2스트라이크)면 존을 넓게 보고 방어적 스윙 -> 스윙 확률 증가
        # 유리한 카운트(3볼 0스트)면 존 좁게 봄 -> 스윙 확률 감소
        strikes = count_context.get('strikes', 0)
        balls = count_context.get('balls', 0)
        
        prob_adj = 0.0
        if strikes == 2: prob_adj += 0.20  # Protect Mode
        if balls == 3: prob_adj -= 0.30    # Wait Mode
        
        # 4. 구종 보정
        # 변화구(SL, CB)는 직구보다 판단 시간이 짧아 스윙 유도 확률이 다름
        if pitch_type in ['SL', 'CB', 'CH']:
            # 유인구(Chase) 위치일 때 스윙 확률 보정
            if dist > 0.8: prob_adj += 0.10 # 잘 속음
            
        final_prob = np.clip(base_prob + prob_adj, 0.01, 0.99)
        return round(final_prob * 100, 1)

    def train_on_local_data(self, batter_df):
        """
        [Advanced] 로컬에 쌓인 타자 데이터가 있다면 즉석에서 RF 모델 학습 (CPU)
        """
        if len(batter_df) < 50: return # 데이터 부족
        
        # Feature: plate_x, plate_z, speed, pfx ...
        # Label: is_swing
        # self.model = RandomForestClassifier(n_estimators=10, n_jobs=-1) # 경량
        # self.model.fit(...)
        # print("⚡ Local Swing Model Trained!")
        pass