import numpy as np
import random
from api.engine.ai_models import GuessHittingModel, SwingTakeModel
from api.engine.sabermetrics import SabermetricsEngine # [New]

class RLStrategyEngine:
    """
    [v7.0 Phase 4] Deep Intelligence Strategy (Context Aware)
    - Guess Hitting + Swing Prob + Leverage Index(LI)
    """
    def __init__(self):
        self.guess_model = GuessHittingModel()
        self.swing_model = SwingTakeModel()
        self.saber_engine = SabermetricsEngine() # [New]
        self.zones = ["High-In", "High-Out", "Low-In", "Low-Out", "Middle"]

    def predict_sequence(self, arsenal, context):
        # 1. 상황 중요도(LI) 계산 [New]
        li = self.saber_engine.calculate_leverage_index(context)
        li_mode = self.saber_engine.get_strategy_mode(li)
        
        # 2. 노림수 및 스윙 확률
        guess_probs = self.guess_model.predict_guess_probabilities(context, arsenal)
        
        # 3. Setup Pitch 결정 (LI 반영)
        setup_res = self._policy_network(context, arsenal, guess_probs, li, li_mode)
        
        # 4. Finish Pitch
        next_ctx = context.copy()
        next_ctx['strikes'] = min(context.get('strikes', 0) + 1, 2)
        next_guess = self.guess_model.predict_guess_probabilities(next_ctx, arsenal)
        # Finish는 LI 영향이 적으므로 기본 LI(1.0) 처리
        finish_res = self._policy_network(next_ctx, arsenal, next_guess, 1.0, "Medium", is_finish=True)
        
        return {
            "strategy_name": f"Context-Aware Strategy ({li_mode})",
            "recommended_pitch": setup_res['pitch'],
            "location": setup_res['loc'],
            "reasoning": setup_res['reasoning'],
            "guess_probs": guess_probs,
            "swing_prob": setup_res['swing_prob'],
            "leverage_index": li, # [New]
            "next_pitch": {"pitch": finish_res['pitch'], "location": finish_res['loc']}
        }

    def _policy_network(self, ctx, arsenal, guess_probs, li, li_mode, is_finish=False):
        b, s = ctx.get('balls', 0), ctx.get('strikes', 0)
        
        # 기본 로직 (Phase 3 유지)
        loc_coords = {"High-In": (-0.7, 3.2), "High-Out": (0.7, 3.2), "Low-In": (-0.7, 1.8), "Low-Out": (0.7, 1.8), "Middle": (0.0, 2.5)}
        
        selected_pitch = "FF"
        selected_loc = "Low-Out"
        reason = ""

        # [Logic Tree with Leverage]
        
        # A. 초위기 상황 (LI > 3.0: Critical) -> 안전 제일 / 구위 중심
        if li > 3.0:
            # 타자의 노림수가 적고, 구종 가치가 높은 공 선택 (변수 창출 자제)
            # 여기서는 단순하게 FF(직구) 위주이나, 실제론 Stuff+ 높은 공을 선택해야 함
            best_pitch = "FF" 
            if "FF" in guess_probs and guess_probs["FF"] > 60:
                # 직구를 노린다면 슬라이더로 도망
                best_pitch = "SL" if "SL" in arsenal else "CH"
            
            selected_pitch = best_pitch
            selected_loc = "Low-Out" # 장타 억제 존
            reason = f"🚨 위기 상황(LI {li})입니다. 장타 위험을 최소화하기 위해 {best_pitch}를 낮게 제구합니다."

        # B. 여유 상황 (LI < 0.7: Garbage) -> 실험 / 과감함
        elif li < 0.7:
            # 평소 안 던지던 공이나 하이존 공략
            selected_pitch = random.choice(arsenal)
            selected_loc = "High-In"
            reason = f"점수 차 여유(LI {li})가 있습니다. {selected_pitch}를 과감하게 몸쪽에 붙여 반응을 확인합니다."

        # C. 일반 상황 (Phase 3 로직)
        else:
            if s == 2:
                if "SL" in arsenal:
                    selected_pitch = "SL"
                    selected_loc = "Low-Out"
                    reason = "결정구 타이밍입니다. 바깥쪽 슬라이더로 유인합니다."
                else:
                    selected_pitch = "FF"
                    selected_loc = "High-In"
                    reason = "하이 패스트볼로 헛스윙을 유도합니다."
            else:
                candidates = [(p, guess_probs.get(p, 0)) for p in arsenal]
                best_pitch = min(candidates, key=lambda x: x[1])[0]
                selected_pitch = best_pitch
                selected_loc = "Low-Out"
                reason = f"타자의 노림수({guess_probs.get(best_pitch)}%)를 피해 카운트를 잡습니다."

        tx, tz = loc_coords.get(selected_loc.split()[0], (0, 2.5))
        swing_p = self.swing_model.predict_swing_prob(selected_pitch, tx, tz, ctx)
        
        return {"pitch": selected_pitch, "loc": selected_loc, "reasoning": reason, "swing_prob": swing_p}