import torch
import joblib
import os
import numpy as np
import pandas as pd
from app.model import PitchLSTM
from app.model_attention import PitchLSTMAttention

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 설정
MODEL_A_PATH = os.path.join(DATA_DIR, "pitch_lstm_global.pth")      # Baseline
MODEL_B_PATH = os.path.join(DATA_DIR, "pitch_lstm_attention.pth")   # Challenger
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.pkl")

def compare_models():
    print("\n⚔️  [Model A: LSTM] vs [Model B: Attention] Showdown  ⚔️\n")

    # 1. 인코더 로드
    if not os.path.exists(ENCODER_PATH):
        print("❌ Encoders not found!")
        return
    encoders = joblib.load(ENCODER_PATH)
    le_pitch = encoders['le_pitch']
    input_size = encoders.get('input_size', 25)
    num_classes = encoders.get('num_classes', 10)

    # 2. Model A (LSTM) 로드
    print(f"📥 Loading Model A (LSTM)...")
    model_a = PitchLSTM(input_size, 128, 2, num_classes).to(DEVICE)
    model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
    model_a.eval()

    # 3. Model B (Attention) 로드 - Mason Miller 전용 모델 우선 탐색
    print(f"📥 Loading Model B (Attention)...")
    model_b = PitchLSTMAttention(input_size, 128, 2, num_classes).to(DEVICE)
    
    # 🌟 [핵심 변경] 개인화된 모델 파일이 있는지 확인하고 로드합니다.
    MILLER_MODEL_PATH = os.path.join(DATA_DIR, "models", "lstm_attention_695243.pth")
    
    if os.path.exists(MILLER_MODEL_PATH):
        print(f"✅ Found Fine-tuned Model for Miller! Loading from: {MILLER_MODEL_PATH}")
        model_b.load_state_dict(torch.load(MILLER_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"⚠️ Fine-tuned model not found. Using Global Base: {MODEL_B_PATH}")
        model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
        
    model_b.eval()

    # 4. 시뮬레이션 상황: Mason Miller vs Aaron Judge (Full Count)
    # 25개 피처를 수동으로 정의 (아까 우리가 정한 값들)
    # [Inning, Balls, Strikes, Outs, ScoreDiff, 1B, 2B, 3B, Stand(R=0), Throw(R=0), ...]
    input_vec = [
        9.0/9.0,   # 9회
        2.0/4.0,   # 2볼
        2.0/3.0,   # 2스트라이크
        2.0/3.0,   # 2아웃
        11.0/20.0, # 1점차
        1.0, 1.0, 1.0, # 만루 (주자 있음)
        0.0, 0.0,  # 우타자, 우투수 (인코딩 값 가정)
        0.15,      # 투구수 15개 (초반)
        0.25,      # 타순 1번째
        0.15,      # 투수 투구수
        0.25, 0.30, # 타자 성향 (Whiff, K)
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, # Z-Scores (평균)
        3.82, 1.75  # ✅ Aaron Judge Zone (높은 존 반영)
    ]
    
    # 길이가 안 맞으면 패딩 (안전장치)
    if len(input_vec) < input_size:
        input_vec.extend([0.0] * (input_size - len(input_vec)))
    
    # 텐서 변환 (시퀀스 길이 5로 복제)
    seq = np.tile(input_vec, (5, 1))
    tensor_x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

    print("\n⚾️ Scenario: Mason Miller vs Aaron Judge (Bases Loaded, Full Count)")
    print("-" * 60)

    # 5. 예측 실행
    with torch.no_grad():
        # Model A 예측
        out_a = model_a(tensor_x)
        prob_a = torch.softmax(out_a, dim=1)[0].cpu().numpy()

        # Model B 예측 (Attention Weights도 반환됨)
        out_b, attn_weights = model_b(tensor_x)
        prob_b = torch.softmax(out_b, dim=1)[0].cpu().numpy()

    # 6. 결과 출력 (Top 3 구종 비교)
    results = []
    for i in range(num_classes):
        pitch_name = le_pitch.inverse_transform([i])[0]
        results.append({
            "Pitch": pitch_name,
            "Model A (%)": round(prob_a[i] * 100, 1),
            "Model B (%)": round(prob_b[i] * 100, 1),
            "Diff": round((prob_b[i] - prob_a[i]) * 100, 1)
        })
    
    # 확률 높은 순으로 정렬 (Model B 기준)
    results.sort(key=lambda x: x['Model B (%)'], reverse=True)

    print(f"{'Pitch Type':<12} | {'Model A (LSTM)':<15} | {'Model B (Attn)':<15} | {'Change':<10}")
    print("-" * 60)
    for res in results[:5]: # Top 5만 출력
        diff_str = f"{res['Diff']:+0.1f}"
        print(f"{res['Pitch']:<12} | {res['Model A (%)']:>5.1f}%          | {res['Model B (%)']:>5.1f}%          | {diff_str:>5}%")
    
    print("-" * 60)
    
    # Attention 가중치 분석
    print("\n🧠 Attention Analysis (Model B Insight):")
    weights = attn_weights[0].cpu().numpy() # (Seq_Len,)
    print(f"Time Step Focus: {weights}")
    print("-> Model B가 5개의 입력 시퀀스 중 어디를 중요하게 봤는지 나타냅니다.")

if __name__ == "__main__":
    compare_models()