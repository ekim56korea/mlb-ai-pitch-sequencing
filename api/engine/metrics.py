import joblib
import pandas as pd
import numpy as np
import os

class MetricsEngine:
    """
    [Phase 2] Pitching+ (Stuff+) Evaluation Engine
    """
    def __init__(self):
        self.model_path = os.path.join("api", "engine", "stuff_plus_model.pkl")
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("✅ Stuff+ AI 모델 로드 완료")
        else:
            print("⚠️ Stuff+ 모델 파일이 없습니다. (학습 필요)")

    def calculate_stuff_plus(self, pitch_data: dict):
        """
        투구 데이터를 받아 Stuff+ 점수(0~200, 평균 100)를 반환
        """
        if self.model is None:
            return 100.0 # Default fallback

        # [수정됨] 모델이 학습할 때 썼던 이름('release_extension')으로 맞춰줘야 합니다.
        features = pd.DataFrame([{
            'release_speed': pitch_data.get('release_speed', 90),
            'release_spin_rate': pitch_data.get('release_spin_rate', 2200),
            'pfx_x': pitch_data.get('pfx_x', 0),
            'pfx_z': pitch_data.get('pfx_z', 0),
            # ▼▼▼ 여기가 수정된 부분입니다 (extension -> release_extension) ▼▼▼
            'release_extension': pitch_data.get('extension', 6.0)
        }])
        
        # 1. xRV (Expected Run Value) 예측
        try:
            predicted_xrv = self.model.predict(features)[0]
        except Exception as e:
            print(f"Prediction Error: {e}")
            return 100.0
        
        # 2. Scaling to "Plus" stat (Mean 100, SD 10 가정)
        # xRV가 낮을수록 좋음 -> 점수는 높아야 함
        avg_xrv = -0.02
        std_dev = 0.05
        
        z_score = (avg_xrv - predicted_xrv) / std_dev
        stuff_plus = 100 + (z_score * 10)
        
        return round(stuff_plus, 1)