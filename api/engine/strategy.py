import random

class StrategyEngine:
    """
    [V4.0] Context-Aware Strategy Engine
    경기 상황(볼카운트, 주자, 아웃)에 따라 최적의 구종과 로케이션을 추천
    """
    def __init__(self):
        # 구종별 기본 특성 (가정)
        self.pitch_specs = {
            "FF": {"desc": "High Fastball", "target": (0.0, 3.5)}, # 하이 패스트볼
            "SL": {"desc": "Low-Away Slider", "target": (0.5, 1.5)}, # 바깥쪽 낮은 슬라이더
            "CH": {"desc": "Low Changeup", "target": (-0.5, 1.5)}, # 몸쪽 낮은 체인지업
            "CB": {"desc": "Low Curve", "target": (0.0, 1.0)}, # 낮게 떨어지는 커브
            "SI": {"desc": "Sinker (Double Play)", "target": (0.0, 1.5)} # 땅볼 유도 싱커
        }

    def recommend_pitch(self, arsenal: list, context: dict):
        """
        상황별 로직 트리 (Decision Tree)
        """
        balls = context.get('balls', 0)
        strikes = context.get('strikes', 0)
        runners = [context.get('runner_on_1b'), context.get('runner_on_2b'), context.get('runner_on_3b')]
        has_runners = any(runners)
        
        # 1. 위기 상황 (주자 있음 + 아웃 카운트 적음) -> 땅볼 유도(Double Play)
        if context.get('runner_on_1b') and context.get('outs', 0) < 2:
            priority = ["SI", "CH", "FS", "FF"] # 싱커/체인지업 우선
            strategy_name = "🚨 Double Play Situation"
            reason = "병살타 유도를 위해 무브먼트가 큰 떨어지는 공을 추천합니다."
        
        # 2. 투수 유리 (2 Strikes) -> 유인구 (Chase)
        elif strikes == 2:
            priority = ["SL", "CB", "FS", "FF"] # 변화구 유인구 우선
            strategy_name = "⚔️ Put Away (Strikeout)"
            reason = "타자가 몰려있습니다. 존 바깥으로 빠지는 변화구로 헛스윙을 유도하세요."
            
        # 3. 타자 유리 (3 Balls) -> 존 공략 (Challenge)
        elif balls == 3:
            priority = ["FF", "SI", "FC"] # 직구 계열 우선
            strategy_name = "🛡️ Challenge Zone"
            reason = "볼넷은 위험합니다. 가장 자신 있는 직구로 존을 공략하세요."
            
        # 4. 초구 또는 일반 상황 -> 카운트 잡기
        else:
            priority = ["FF", "SI", "SL", "CH"]
            strategy_name = "🎯 Get Ahead"
            reason = "유리한 카운트를 선점하기 위해 초구 스트라이크를 잡으세요."

        # 투수가 던질 수 있는 구종 중 우선순위가 높은 것 선택
        best_pitch = "FF" # Default
        for p in priority:
            if p in arsenal:
                best_pitch = p
                break
        
        spec = self.pitch_specs.get(best_pitch, {"desc": "Standard", "target": (0.0, 2.5)})
        
        return {
            "recommended_pitch": best_pitch,
            "location_desc": spec['desc'],
            "target_x": spec['target'][0],
            "target_z": spec['target'][1],
            "strategy_name": strategy_name,
            "reasoning": reason
        }