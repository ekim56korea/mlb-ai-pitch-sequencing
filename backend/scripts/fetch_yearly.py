from pybaseball import statcast
import pandas as pd
import os
import time

# 설정
START_YEAR = 2025
END_YEAR = 2025
DATA_DIR = "savant_data"

def fetch_data_by_year():
    # 폴더가 없으면 생성
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created directory: {DATA_DIR}")

    print(f"⚾️ Starting partitioned download ({START_YEAR}-{END_YEAR})...")

    for year in range(START_YEAR, END_YEAR + 1):
        file_path = os.path.join(DATA_DIR, f"{year}.csv")
        
        # 🌟 이미 다운로드 받은 파일이 있으면 건너뜀 (이어받기 기능)
        if os.path.exists(file_path):
            print(f"✅ {year} already exists. Skipping...")
            continue

        print(f"\n⬇️ Downloading {year} season...")
        
        try:
            start_time = time.time()
            # 3월 1일 ~ 11월 30일 (넉넉하게 잡음)
            df = statcast(start_dt=f"{year}-03-01", end_dt=f"{year}-11-30")
            
            if not df.empty:
                # CSV로 즉시 저장
                df.to_csv(file_path, index=False)
                elapsed = time.time() - start_time
                print(f"   💾 Saved {year}.csv ({len(df):,} pitches) - {elapsed:.1f}s")
            else:
                print(f"   ⚠️ No data found for {year}")

        except Exception as e:
            print(f"   ❌ Error downloading {year}: {e}")
            # 에러가 나도 다음 연도로 넘어갑니다.

    print("\n🎉 All downloads finished!")
    print(f"📂 Check the '{DATA_DIR}' folder.")

if __name__ == "__main__":
    fetch_data_by_year()