import numpy as np

class SabermetricsEngine:
    """
    [v7.0 Phase 4] Sabermetrics Engine (Zero-Cost Context)
    - RE24 (Run Expectancy Matrix) 기반 상황 중요도 계산
    - Leverage Index (LI) 산출
    """
    def __init__(self):
        # [RE24 Matrix] 2010-2020 MLB 평균 기대 득점 (Base-Out State)
        # Key: (Outs, 1B, 2B, 3B) -> Value: Expected Runs
        self.re24_matrix = {
            (0, 0, 0, 0): 0.48, (0, 1, 0, 0): 0.85, (0, 0, 1, 0): 1.07, (0, 0, 0, 1): 1.35,
            (0, 1, 1, 0): 1.44, (0, 1, 0, 1): 1.67, (0, 0, 1, 1): 1.94, (0, 1, 1, 1): 2.30,
            
            (1, 0, 0, 0): 0.25, (1, 1, 0, 0): 0.50, (1, 0, 1, 0): 0.66, (1, 0, 0, 1): 0.94,
            (1, 1, 1, 0): 0.90, (1, 1, 0, 1): 1.13, (1, 0, 1, 1): 1.37, (1, 1, 1, 1): 1.54,
            
            (2, 0, 0, 0): 0.10, (2, 1, 0, 0): 0.21, (2, 0, 1, 0): 0.31, (2, 0, 0, 1): 0.36,
            (2, 1, 1, 0): 0.42, (2, 1, 0, 1): 0.49, (2, 0, 1, 1): 0.56, (2, 1, 1, 1): 0.76
        }
        
        # 기본 레버리지 (1.0 = Average)
        self.avg_li = 1.0

    def calculate_run_expectancy(self, outs, r1, r2, r3):
        """현재 상황의 기대 득점 반환"""
        key = (outs, int(r1), int(r2), int(r3))
        return self.re24_matrix.get(key, 0.0)

    def calculate_leverage_index(self, context):
        """
        경기 중요도(LI) 계산
        - Inning, Score Diff, Runners, Outs를 고려
        - Zero-Cost 근사식: LI ~ (BaseVolaity * InningFactor) / (ScoreDiffFactor)
        """
        inning = context.get('inning', 1)
        score_diff = context.get('score_diff', 0) # Pitcher Team - Batter Team
        outs = context.get('outs', 0)
        r1, r2, r3 = context.get('runner_on_1b'), context.get('runner_on_2b'), context.get('runner_on_3b')
        
        # 1. Base State Volatility (주자가 많을수록 위기)
        # RE24 값을 이용해 현재 상황의 변동성 측정
        re_current = self.calculate_run_expectancy(outs, r1, r2, r3)
        base_factor = 1.0 + re_current
        
        # 2. Inning Factor (경기 후반일수록 중요)
        # 1~6회: 1.0, 7~8회: 1.5, 9회+: 2.5
        inn_factor = 1.0
        if inning >= 9: inn_factor = 2.5
        elif inning >= 7: inn_factor = 1.5
        
        # 3. Score Difference Factor (점수차가 적을수록 중요)
        # 0점차: 극대화, 4점차 이상: 급격히 감소
        abs_diff = abs(score_diff)
        score_factor = 1.0
        if abs_diff == 0: score_factor = 2.0
        elif abs_diff == 1: score_factor = 1.6
        elif abs_diff == 2: score_factor = 1.2
        elif abs_diff >= 4: score_factor = 0.5 # Garbage Time
        
        # 최종 LI 계산 (근사치)
        li = base_factor * inn_factor * score_factor / 2.0 # Normalize approx
        
        # Clipping (0.0 ~ 10.0)
        return round(min(max(li, 0.1), 10.0), 2)

    def get_strategy_mode(self, li):
        """LI에 따른 전략 모드 반환"""
        if li > 3.0: return "Critical (High-Leverage)"
        elif li > 1.5: return "High"
        elif li < 0.7: return "Low (Garbage Time)"
        else: return "Medium"