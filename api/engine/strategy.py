import random

class StrategyEngine:
    """
    SRS REQ-STR-01, 02: 게임 이론 및 xRV 기반 투구 추천 엔진
    """
    def __init__(self):
        # xRV (Expected Run Value) 매트릭스 (가상의 데이터)
        # 행: 타자 클러스터 ID, 열: 구종 및 결과
        # 값이 낮을수록 투수에게 유리 (실점 확률 감소)
        self.xrv_matrix = {
            0: {"FF": 0.05, "SL": -0.02, "CB": 0.01}, # 공격적 컨택터 -> 슬라이더 유리
            1: {"FF": -0.05, "SL": 0.03, "CB": 0.02}, # 신중형 -> 직구로 카운트 잡기 유리
            2: {"FF": 0.10, "SL": 0.08, "CB": 0.05},  # 선구안 -> 그나마 커브가 나음
            3: {"FF": -0.02, "SL": -0.15, "CB": -0.10}, # 공풍기 -> 변화구 대박 유리
            4: {"FF": 0.00, "SL": -0.20, "CB": -0.05}   # 배드볼 히터 -> 슬라이더 유인구 최강
        }

    def recommend_pitch(self, cluster_id: int, ball_count: str):
        """
        타자 성향(Cluster)과 볼카운트를 고려하여 최적의 구종을 추천합니다.
        """
        # 1. 해당 타자에게 유리한 구종들의 xRV 가져오기
        options = self.xrv_matrix.get(cluster_id, {"FF": 0, "SL": 0, "CB": 0})
        
        # 2. 볼카운트 상황 보정 (Heuristic Rule)
        # 예: 3-0, 3-1 불리한 카운트에서는 직구(FF) 비율 높임
        if ball_count in ["3-0", "3-1"]:
            options["FF"] -= 0.1  # 직구의 가치를 높임 (강제성)

        # 2스트라이크 이후에는 유인구(SL, CB) 가치 높임
        if "2S" in ball_count or ball_count == "0-2":
            options["SL"] -= 0.05
            options["CB"] -= 0.05

        # 3. 최적 전략 선정 (가장 낮은 xRV를 가진 구종 찾기)
        # REQ-STR-02: Entropy Mixing (패턴 노출 방지를 위해 1등만 뽑지 않고 확률적으로 섞음)
        # 여기서는 간단하게 가장 좋은 구종을 메인으로 추천
        best_pitch = min(options, key=options.get)
        min_xrv = options[best_pitch]
        
        # 추천 사유 생성
        reasoning = f"타자 유형(ID:{cluster_id}) 상대로 '{best_pitch}'의 기대 실점(xRV)이 {min_xrv}로 가장 낮음."
        
        return {
            "recommended_pitch": best_pitch,
            "xrv_score": min_xrv,
            "reasoning": reasoning,
            "mix_strategy": options # 전체 옵션 점수
        }