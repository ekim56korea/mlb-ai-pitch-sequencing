# check_files.py (수정 버전)
import torch
import joblib
import os
import sys

# Docker 내부에서는 /code/app/check_files.py 이므로, /code/data로 가야 합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data") # <-- BASE_DIR의 상위 디렉토리로 이동 후 data 폴더 접근

MODEL_PATH = os.path.join(DATA_DIR, "pitch_lstm.pth")
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.pkl")

def inspect_files():
    print(f"📂 데이터 폴더 위치: {DATA_DIR}")
    print("=== 🔍 파일 검증 시작 ===")

    # 1. Encoders 확인
    if os.path.exists(ENCODER_PATH):
        print(f"\n✅ encoders.pkl 발견! (크기: {os.path.getsize(ENCODER_PATH)} bytes)")
        try:
            encoders = joblib.load(ENCODER_PATH)
            classes = encoders['le_pitch'].classes_
            print(f"⚾️ 학습된 구종 개수: {len(classes)}개")
            print(f"📝 구종 목록: {classes}")
            
            scaler = encoders['scaler']
            # 데이터 범위가 0~1 사이인지 확인 (스케일링 여부)
            print(f"📏 스케일러 샘플(Min): {scaler.data_min_[:3]}") 
        except Exception as e:
             print(f"⚠️ 인코더 로드 중 오류: {e}")
    else:
        print(f"❌ encoders.pkl 파일이 없습니다. (경로: {ENCODER_PATH})")

    # 2. LSTM 모델 확인
    if os.path.exists(MODEL_PATH):
        print(f"\n✅ pitch_lstm.pth 발견! (크기: {os.path.getsize(MODEL_PATH)} bytes)")
        
        try:
            # CPU로 강제 로드 (GPU 없어도 되게)
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            
            # 가중치 텐서 확인
            keys = list(state_dict.keys())
            print(f"🧠 모델 레이어 개수: {len(keys)}개")
            
            # 첫 번째 레이어의 크기 분석
            if 'lstm.weight_ih_l0' in state_dict:
                weight_shape = state_dict['lstm.weight_ih_l0'].shape
                print(f"📐 입력 레이어 크기: {weight_shape}")
                print("   (이 크기가 (512, 13) 정도라면 13개 피처가 정상 반영된 것입니다.)")
            
            print("\n🎉 결론: 파일이 정상적으로 생성되었고 학습 내용이 담겨 있습니다!")
            
        except Exception as e:
            print(f"❌ 모델 파일은 있지만 로드에 실패했습니다: {e}")
    else:
        print(f"❌ pitch_lstm.pth 파일이 없습니다. (경로: {MODEL_PATH})")

if __name__ == "__main__":
    inspect_files()