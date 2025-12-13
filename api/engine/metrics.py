import joblib
import pandas as pd
import numpy as np
import os
from api.engine.preprocessor import DataPreprocessor

class MetricsEngine:
    """
    [v6.0 Final] Metrics Engine
    - Stuff+ (with D-Plane & Feature Engineering)
    - Location+ (with Context)
    - Pitching+ (Weighted)
    - Fatigue Decay & Tunneling Bonus
    """
    def __init__(self):
        self.model_dir = os.path.join("api", "engine")
        self.stuff_model = None
        self.loc_model = None
        self.preprocessor = DataPreprocessor()
        self.load_models()

    def load_models(self):
        stuff_path = os.path.join(self.model_dir, "stuff_plus_model.pkl")
        if os.path.exists(stuff_path): self.stuff_model = joblib.load(stuff_path)
        loc_path = os.path.join(self.model_dir, "location_plus_model.pkl")
        if os.path.exists(loc_path): self.loc_model = joblib.load(loc_path)

    def _scale_to_plus(self, predicted_xrv, avg_xrv=-0.02, std_dev=0.05):
        z_score = (avg_xrv - predicted_xrv) / std_dev
        return 100 + (z_score * 10)

    def calculate_d_plane_effect(self, pfx_x, pfx_z, pitch_type):
        """D-Plane Adjustment"""
        effect_score = 0.0
        if pitch_type in ['FF', 'FC']:
            if pfx_z > 15: effect_score -= 2.0 
        elif pitch_type in ['SI', 'CH', 'FS']:
            if pfx_z < 5: effect_score += 3.0 
        if abs(pfx_x) > 12: effect_score += 2.0
        return effect_score

    def calculate_fatigue_penalty(self, pitch_count):
        """Fatigue Decay Function"""
        if pitch_count < 80: return 0.0
        over_count = pitch_count - 80
        penalty = 0.1 * (over_count ** 1.5)
        return -min(penalty, 20.0)

    def calculate_tunneling_bonus(self, current_pos, prev_pos):
        """Tunneling Bonus at Commit Point"""
        if current_pos is None or prev_pos is None: return 0.0
        dx = current_pos[0] - prev_pos[0]
        dz = current_pos[1] - prev_pos[1]
        dist = np.sqrt(dx**2 + dz**2)
        
        if dist < 0.5: return 5.0
        elif dist < 1.0: return 2.0
        elif dist < 1.5: return 0.0
        else: return -2.0

    def calculate_pitching_plus(self, pitch_data: dict, context: dict = None):
        if context is None: context = {'balls': 0, 'strikes': 0, 'stand': 'R'}

        # 1. Stuff+
        stuff_score = 100.0
        pred_stuff_xrv = 0.0
        if self.stuff_model:
            try:
                raw_df = pd.DataFrame([{
                    'release_speed': pitch_data.get('release_speed', 90),
                    'release_spin_rate': pitch_data.get('release_spin_rate', 2200),
                    'pfx_x': pitch_data.get('pfx_x', 0),
                    'pfx_z': pitch_data.get('pfx_z', 0),
                    'release_extension': pitch_data.get('extension', 6.0)
                }])
                engineered_df = self.preprocessor.engineer_features(raw_df)
                features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_extension',
                            'effective_velo', 'velo_pfx_z_interaction', 'movement_per_spin']
                
                pred_stuff_xrv = self.stuff_model.predict(engineered_df[features])[0]
                stuff_score = self._scale_to_plus(pred_stuff_xrv)
                
                # D-Plane
                stuff_score += self.calculate_d_plane_effect(
                    pitch_data.get('pfx_x', 0), pitch_data.get('pfx_z', 0), pitch_data.get('pitch_type', 'FF')
                )
            except Exception as e: print(f"Stuff Error: {e}")

        # Fatigue & Tunneling
        fatigue_pen = self.calculate_fatigue_penalty(pitch_data.get('pitch_count', 0))
        stuff_score += fatigue_pen
        
        tunnel_bonus = pitch_data.get('tunneling_bonus', 0.0)
        stuff_score += tunnel_bonus

        # 2. Location+
        loc_score = 100.0
        if self.loc_model:
            try:
                stand_code = 1 if context.get('stand') == 'L' else 0
                loc_feats = pd.DataFrame([{
                    'plate_x': pitch_data.get('plate_x', 0.0),
                    'plate_z': pitch_data.get('plate_z', 2.5),
                    'balls': context.get('balls', 0),
                    'strikes': context.get('strikes', 0),
                    'stand_code': stand_code
                }])
                pred_loc_xrv = self.loc_model.predict(loc_feats)[0]
                loc_score = self._scale_to_plus(pred_loc_xrv)
            except Exception as e: print(f"Loc Error: {e}")

        # 3. Pitching+
        pitching_plus = (stuff_score * 0.5) + (loc_score * 0.5)

        return {
            "stuff_plus": round(stuff_score, 1),
            "location_plus": round(loc_score, 1),
            "pitching_plus": round(pitching_plus, 1),
            "xRV": round(pred_stuff_xrv, 4),
            "details": {
                "fatigue_penalty": round(fatigue_pen, 1),
                "tunneling_bonus": round(tunnel_bonus, 1)
            }
        }

    def get_adjusted_score(self, raw_score, sample_size):
        return self.preprocessor.apply_bayesian_shrinkage(raw_score, sample_size)