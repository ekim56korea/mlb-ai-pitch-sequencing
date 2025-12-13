import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

class StrategyTransferEngine:
    """
    [v6.0 Phase 2] Nearest Neighbor Strategy Transfer
    데이터가 부족한 투수를 위해 유사한 스타일의 MLB 베테랑 투수(Archetype)를 찾아
    그들의 성공 전략(구종 배합, 핫존)을 전이(Transfer)합니다.
    """
    def __init__(self):
        # MLB 대표 투수들의 아키타입 데이터 (예시 하드코딩 + 확장 가능성)
        # 실제로는 DB에서 전체 투수 평균 데이터를 쿼리해와야 함
        self.archetypes = pd.DataFrame([
            {'name': 'Gerrit Cole', 'velo': 96.7, 'pfx_x': -8.0, 'pfx_z': 10.0, 'type': 'Power Pitcher'},
            {'name': 'Clayton Kershaw', 'velo': 90.7, 'pfx_x': 2.0, 'pfx_z': 8.0, 'type': 'Crafty Lefty'},
            {'name': 'Jacob deGrom', 'velo': 99.0, 'pfx_x': -6.0, 'pfx_z': 11.0, 'type': 'Elite Fastball'},
            {'name': 'Greg Maddux (Sim)', 'velo': 88.0, 'pfx_x': -10.0, 'pfx_z': 4.0, 'type': 'Control Artist'},
            {'name': 'Shohei Ohtani', 'velo': 97.0, 'pfx_x': -9.0, 'pfx_z': 9.0, 'type': 'Unicorn'},
            {'name': 'Devin Williams', 'velo': 94.0, 'pfx_x': -12.0, 'pfx_z': -4.0, 'type': 'Changeup Specialist'}
        ])
        
        # KNN 모델 학습 (Feature: 구속, 수평무브먼트, 수직무브먼트)
        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(self.archetypes[['velo', 'pfx_x', 'pfx_z']])

    def find_similar_pitcher(self, velocity, pfx_x, pfx_z):
        """
        입력된 투구 특성과 가장 유사한 베테랑 투수를 찾습니다.
        """
        query = np.array([[velocity, pfx_x, pfx_z]])
        dist, idx = self.knn.kneighbors(query)
        
        similar_pitcher = self.archetypes.iloc[idx[0][0]]
        similarity_score = 100 - (dist[0][0] * 2) # 거리 기반 점수화 (단순화)
        
        return {
            "name": similar_pitcher['name'],
            "type": similar_pitcher['type'],
            "similarity": round(max(0, similarity_score), 1)
        }