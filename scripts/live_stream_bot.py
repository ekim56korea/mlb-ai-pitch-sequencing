import pandas as pd
import requests
import time
import os
import random

# 서버 주소
API_URL = "http://127.0.0.1:8000/live/ingest"

def start_streaming():
    print("⚾️ MLB 라이브 스트리밍 봇을 가동합니다...")
    
    # 1. 데이터 로드
    data_path = os.path.join("data", "statcast_sample.parquet")
    if not os.path.exists(data_path):
        print("❌ 데이터 파일이 없습니다.")
        return
        
    df = pd.read_parquet(data_path)
    # 시뮬레이션을 위해 무작위로 섞음
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"📦 총 {len(df)}개의 투구 데이터를 순차 전송합니다.\n")
    
    # 2. 데이터 전송 루프
    for i, row in df.iterrows():
        # 보낼 데이터 포맷팅 (Pydantic 스키마에 맞춤)
        payload = {
            "pitch_type": row.get('pitch_type', 'FF'),
            "release_speed": float(row.get('release_speed', 90.0)),
            "release_spin_rate": float(row.get('release_spin_rate', 2000.0))
            # 필요한 경우 9-param 등 추가 필드 확장 가능
        }
        
        try:
            # POST 요청으로 데이터 전송
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"[{i+1}/{len(df)}] 🚀 전송 성공: {payload['pitch_type']} {payload['release_speed']}mph")
            else:
                print(f"❌ 전송 실패: {response.text}")
                
        except Exception as e:
            print(f"⚠️ 연결 오류: 서버가 켜져 있나요? ({e})")
            break
            
        # 3. 실제 경기처럼 대기 (1초 ~ 3초 랜덤)
        time.sleep(random.uniform(1.0, 3.0))

if __name__ == "__main__":
    start_streaming()