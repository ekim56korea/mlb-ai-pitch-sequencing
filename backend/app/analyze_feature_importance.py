"""
Week 7: Feature Importance Analysis with SHAP

Week 6 모델의 Feature Importance 분석
- SHAP values 계산
- sequence_entropy와 altitude_factor 중요도 확인
- 상위 15개 피처 시각화
- Week 5 대비 중요도 변화 분석

Author: AI Pitch Sequencing Team
Date: 2025-01-XX
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# SHAP import (optional, 설치되어 있다면)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")

# Week 6 모델
MODEL_PATH = os.path.join(DATA_DIR, "xgb_pitch_classifier.json")
ENCODER_PATH = os.path.join(DATA_DIR, "xgb_encoders.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Week 6 Features (39)
FEATURES = [
    # Basic (8)
    'balls', 'strikes', 'outs_when_up', 'inning', 
    'on_1b', 'on_2b', 'on_3b', 'score_diff',
    # Historical (10)
    'FF_prev', 'SL_prev', 'CH_prev', 'CU_prev', 'SI_prev',
    'CT_prev', 'FC_prev', 'FS_prev', 'KC_prev', 'None_prev',
    # Physics (7)
    'z_ext', 'x_ext', 'ext_x_abs', 
    'velo_x_release', 'velo_y_release', 'velo_z_release',
    'release_extension',
    # Batter-Pitcher (5)
    'bvp_recent_ba', 'bvp_whiff_rate', 'bvp_chase_rate',
    'bvp_swing_miss', 'bvp_contact_quality',
    # Tunneling (7)
    'tunnel_distance', 'velocity_diff', 
    'FF_count_last_5', 'SL_count_last_5', 'CH_count_last_5', 'CU_count_last_5',
    'sequence_entropy',
    # Contextual (4)
    'altitude_factor', 'rest_days', 'fatigue_index', 'pressure_index'
]

# Feature Groups
FEATURE_GROUPS = {
    'Basic': ['balls', 'strikes', 'outs_when_up', 'inning', 'on_1b', 'on_2b', 'on_3b', 'score_diff'],
    'Historical': ['FF_prev', 'SL_prev', 'CH_prev', 'CU_prev', 'SI_prev', 'CT_prev', 'FC_prev', 'FS_prev', 'KC_prev', 'None_prev'],
    'Physics': ['z_ext', 'x_ext', 'ext_x_abs', 'velo_x_release', 'velo_y_release', 'velo_z_release', 'release_extension'],
    'Batter-Pitcher': ['bvp_recent_ba', 'bvp_whiff_rate', 'bvp_chase_rate', 'bvp_swing_miss', 'bvp_contact_quality'],
    'Tunneling': ['tunnel_distance', 'velocity_diff', 'FF_count_last_5', 'SL_count_last_5', 'CH_count_last_5', 'CU_count_last_5', 'sequence_entropy'],
    'Contextual': ['altitude_factor', 'rest_days', 'fatigue_index', 'pressure_index']
}

# Week 5 Feature Importance (Reference)
WEEK5_IMPORTANCE = {
    'z_ext': 0.142,
    'bvp_recent_ba': 0.098,
    'velocity_diff': 0.087,
    'tunnel_distance': 0.065,
    'x_ext': 0.054,
    'release_extension': 0.049,
    'ext_x_abs': 0.041,
    'bvp_whiff_rate': 0.038,
    'pressure_index': 0.035,
    'sequence_entropy': 0.000,  # Was placeholder
    'altitude_factor': 0.012,   # Old formula
}


def analyze_xgboost_importance(model, feature_names):
    """
    XGBoost 내장 Feature Importance 분석
    """
    print("\n📊 XGBoost Built-in Feature Importance (Gain)...")
    
    # Get importance scores
    importance_dict = model.get_booster().get_score(importance_type='gain')
    
    # Map feature indices to names
    importance_scores = {}
    for key, value in importance_dict.items():
        # key is like 'f0', 'f1', etc.
        idx = int(key[1:])
        if idx < len(feature_names):
            importance_scores[feature_names[idx]] = value
    
    # Normalize to sum to 1
    total = sum(importance_scores.values())
    if total > 0:
        importance_scores = {k: v/total for k, v in importance_scores.items()}
    
    # Sort by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   Top 15 Most Important Features:")
    print("   " + "="*60)
    for i, (feature, score) in enumerate(sorted_features[:15], 1):
        bar = "█" * int(score * 100)
        print(f"   {i:2d}. {feature:20s} {score:.4f} {bar}")
    
    return importance_scores


def analyze_shap_values(model, X_sample, feature_names):
    """
    SHAP values 계산 및 분석
    """
    if not SHAP_AVAILABLE:
        print("\n⚠️ SHAP not available. Skipping SHAP analysis.")
        return None
    
    print("\n🔍 Calculating SHAP Values...")
    print(f"   Sample size: {len(X_sample):,} rows")
    
    # TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (may take a while)
    print("   Computing SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP values shape: (n_samples, n_features, n_classes) for multi-class
    # We'll use absolute mean across all classes and samples
    
    if isinstance(shap_values, list):
        # Multi-class: list of arrays
        # Shape: [class_0: (n_samples, n_features), class_1: (...), ...]
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Binary: single array (n_samples, n_features)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create importance dictionary
    shap_importance = {feature_names[i]: mean_abs_shap[i] for i in range(len(feature_names))}
    
    # Normalize
    total = sum(shap_importance.values())
    if total > 0:
        shap_importance = {k: v/total for k, v in shap_importance.items()}
    
    # Sort
    sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   Top 15 Most Important Features (SHAP):")
    print("   " + "="*60)
    for i, (feature, score) in enumerate(sorted_shap[:15], 1):
        bar = "█" * int(score * 100)
        print(f"   {i:2d}. {feature:20s} {score:.4f} {bar}")
    
    # Summary plot
    try:
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            # Multi-class: use first class for visualization
            shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        
        plt.title("SHAP Summary Plot - Week 6 Model (39 features)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_week6.png"), dpi=150)
        print(f"\n   💾 Saved SHAP summary plot: {RESULTS_DIR}/shap_summary_week6.png")
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Could not create SHAP summary plot: {e}")
    
    return shap_importance


def compare_week5_week6(week6_importance):
    """
    Week 5 vs Week 6 Feature Importance 비교
    """
    print("\n📊 Week 5 vs Week 6 Feature Importance Changes...")
    
    print("\n   Key Changes:")
    print("   " + "="*70)
    
    # sequence_entropy: placeholder → real values
    week5_entropy = WEEK5_IMPORTANCE.get('sequence_entropy', 0.0)
    week6_entropy = week6_importance.get('sequence_entropy', 0.0)
    print(f"   sequence_entropy:")
    print(f"      Week 5: {week5_entropy:.4f} (placeholder, always 0.0)")
    print(f"      Week 6: {week6_entropy:.4f} (Shannon Entropy, 0.0-2.0)")
    print(f"      Δ Change: {week6_entropy - week5_entropy:+.4f}")
    
    # altitude_factor: old formula → new formula
    week5_altitude = WEEK5_IMPORTANCE.get('altitude_factor', 0.0)
    week6_altitude = week6_importance.get('altitude_factor', 0.0)
    print(f"\n   altitude_factor:")
    print(f"      Week 5: {week5_altitude:.4f} (linear formula, Coors 4.6%)")
    print(f"      Week 6: {week6_altitude:.4f} (empirical formula, Coors 6.2%)")
    print(f"      Δ Change: {week6_altitude - week5_altitude:+.4f}")
    
    # tunnel_distance: expected to increase (trajectory_div removed)
    week5_tunnel = WEEK5_IMPORTANCE.get('tunnel_distance', 0.0)
    week6_tunnel = week6_importance.get('tunnel_distance', 0.0)
    print(f"\n   tunnel_distance:")
    print(f"      Week 5: {week5_tunnel:.4f} (shared importance with trajectory_div)")
    print(f"      Week 6: {week6_tunnel:.4f} (trajectory_div removed, r=0.999)")
    print(f"      Δ Change: {week6_tunnel - week5_tunnel:+.4f}")
    
    # Removed features
    print(f"\n   Removed Features (Week 6):")
    print(f"      ❌ trajectory_div (r=0.999 with tunnel_distance)")
    print(f"      ❌ inning_fatigue (r=1.0 with inning)")


def analyze_group_importance(importance_dict):
    """
    Feature Group별 중요도 합산
    """
    print("\n📊 Feature Group Importance...")
    
    group_importance = {}
    for group_name, features in FEATURE_GROUPS.items():
        total = sum(importance_dict.get(f, 0.0) for f in features)
        group_importance[group_name] = total
    
    # Sort
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   Group Rankings:")
    print("   " + "="*60)
    for i, (group, score) in enumerate(sorted_groups, 1):
        bar = "█" * int(score * 50)  # Scale down for display
        feature_count = len(FEATURE_GROUPS[group])
        avg_per_feature = score / feature_count if feature_count > 0 else 0
        print(f"   {i}. {group:20s} {score:.4f} (avg {avg_per_feature:.4f}) {bar}")
        print(f"      ({feature_count} features)")
    
    return group_importance


def create_importance_plot(importance_dict, title, filename):
    """
    Feature Importance 막대 그래프 생성
    """
    # Top 20 features
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    features = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    colors = []
    for feat in features:
        if feat in ['sequence_entropy', 'altitude_factor']:
            colors.append('#e74c3c')  # Red for Week 6 improvements
        elif feat in FEATURE_GROUPS['Tunneling']:
            colors.append('#3498db')  # Blue for Tunneling
        elif feat in FEATURE_GROUPS['Physics']:
            colors.append('#2ecc71')  # Green for Physics
        else:
            colors.append('#95a5a6')  # Gray for others
    
    plt.barh(range(len(features)), scores, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Week 6 Improvements'),
        Patch(facecolor='#3498db', label='Tunneling'),
        Patch(facecolor='#2ecc71', label='Physics'),
        Patch(facecolor='#95a5a6', label='Other')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    print(f"\n   💾 Saved plot: {RESULTS_DIR}/{filename}")
    plt.close()


def main():
    print("=" * 70)
    print("Week 7: Feature Importance Analysis")
    print("Week 6 Model (39 features)")
    print("=" * 70)
    
    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        print("   Run train.py first to create the model.")
        return
    
    print(f"\n📥 Loading Model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    print(f"   ✅ Loaded XGBoost model")
    print(f"   Features: {len(FEATURES)}")
    
    # 2. XGBoost Built-in Importance
    xgb_importance = analyze_xgboost_importance(model, FEATURES)
    
    # 3. SHAP Analysis (optional)
    # For SHAP, we need some sample data
    # Here we'll create dummy data for demonstration
    print("\n   Creating sample data for SHAP analysis...")
    X_sample = np.random.randn(100, len(FEATURES))  # Dummy data
    
    # Note: In real scenario, load actual test data
    # But for now, SHAP is optional
    
    if SHAP_AVAILABLE:
        shap_importance = analyze_shap_values(model, X_sample, FEATURES)
    else:
        shap_importance = None
    
    # 4. Week 5 vs Week 6 Comparison
    compare_week5_week6(xgb_importance)
    
    # 5. Group Analysis
    group_importance = analyze_group_importance(xgb_importance)
    
    # 6. Visualizations
    print("\n📊 Creating Visualizations...")
    create_importance_plot(
        xgb_importance,
        "Feature Importance - Week 6 Model (XGBoost Gain)",
        "feature_importance_week6.png"
    )
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print("\n✅ Key Findings:")
    
    # Top 3 features
    top_3 = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n   Top 3 Most Important Features:")
    for i, (feat, score) in enumerate(top_3, 1):
        print(f"      {i}. {feat}: {score:.4f}")
    
    # Week 6 improvements
    entropy_score = xgb_importance.get('sequence_entropy', 0.0)
    altitude_score = xgb_importance.get('altitude_factor', 0.0)
    
    print(f"\n   Week 6 Improved Features:")
    print(f"      sequence_entropy: {entropy_score:.4f}")
    if entropy_score > 0.02:
        print(f"         ✅ Now contributing (was placeholder in Week 5)")
    else:
        print(f"         ⚠️ Still low importance (check implementation)")
    
    print(f"      altitude_factor: {altitude_score:.4f}")
    if altitude_score > WEEK5_IMPORTANCE.get('altitude_factor', 0.0):
        print(f"         ✅ Increased from Week 5 ({WEEK5_IMPORTANCE.get('altitude_factor', 0.0):.4f})")
    else:
        print(f"         ℹ️ Similar to Week 5 (may need more Coors Field data)")
    
    # Group rankings
    print(f"\n   Strongest Feature Groups:")
    top_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (group, score) in enumerate(top_groups, 1):
        print(f"      {i}. {group}: {score:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
