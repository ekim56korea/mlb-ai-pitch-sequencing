import os
import pandas as pd
from pybaseball import statcast

# 저장할 폴더 설정 (data 폴더)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True) # 폴더가 없으면 생성

def fetch_sample_data():
    print("⚾️ Statcast 데이터를 다운로드합니다... (약 1분 소요)")
    
    # 2024년 8월 1일부터 8월 2일까지의 데이터만 샘플로 가져옵니다.
    # pybaseball은 인터넷을 통해 MLB 서버에서 데이터를 긁어옵니다.
    data = statcast(start_dt='2024-08-01', end_dt='2024-08-02')
    
    # 데이터가 정상적으로 왔는지 확인
    if data is not None and not data.empty:
        print(f"✅ 데이터 다운로드 성공! 총 {len(data)}개의 투구 데이터를 가져왔습니다.")
        
        # 파일로 저장 (Parquet 형식이 CSV보다 빠르고 용량이 작습니다)
        save_path = os.path.join(DATA_DIR, 'statcast_sample.parquet')
        data.to_parquet(save_path, engine='pyarrow') # pandas 설치시 pyarrow도 보통 같이 설치됨
        
        print(f"💾 데이터가 저장되었습니다: {save_path}")
        
        # 데이터 미리보기 (상위 5개 행)
        print("\n[데이터 미리보기]")
        print(data[['player_name', 'pitch_type', 'release_speed', 'release_spin_rate']].head())
    else:
        print("❌ 데이터를 가져오지 못했습니다.")

if __name__ == "__main__":
    fetch_sample_data()