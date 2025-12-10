import pandas as pd
from pybaseball import statcast
import sqlite3
import os
from datetime import datetime, timedelta
import time
from tqdm import tqdm # 진행률 표시바 (pip install tqdm 필요)

# --- 설정 ---
START_YEAR = 2022
END_YEAR = 2025
DB_PATH = os.path.join("data", "mlb_statcast.db")

def init_db():
    """데이터베이스 및 테이블 초기화"""
    # data 폴더가 없으면 생성
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 쿼리 속도를 높이기 위한 인덱스 생성 (테이블이 생성된 후 적용됨)
    # 여기서는 연결만 확인하고 종료
    conn.close()
    print(f"🗄️ 데이터베이스 경로 설정 완료: {DB_PATH}")

def generate_date_ranges(start_year, end_year):
    """
    1년 치를 한 번에 받으면 터지므로, 1주일 단위(Weekly) 날짜 구간을 생성합니다.
    """
    start_date = datetime(start_year, 3, 1) # 보통 3월부터 시즌 시작
    end_date = datetime(end_year, 11, 30)   # 11월이면 시즌 종료
    
    current_date = start_date
    ranges = []
    
    while current_date < end_date:
        next_date = current_date + timedelta(days=6) # 7일 간격
        
        # 시즌 기간(3월~11월)만 포함 (비시즌 데이터 요청 방지)
        if 3 <= current_date.month <= 11:
            ranges.append((current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')))
        
        current_date = next_date + timedelta(days=1)
        
    return ranges

def download_and_store():
    """메인 로직: 구간별 다운로드 -> DB 적재"""
    init_db()
    
    date_ranges = generate_date_ranges(START_YEAR, END_YEAR)
    print(f"📦 총 {len(date_ranges)}개의 주간 데이터 블록을 다운로드합니다. (예상 소요시간: 30분~1시간)")
    
    conn = sqlite3.connect(DB_PATH)
    
    total_rows = 0
    
    # tqdm으로 진행률 표시
    pbar = tqdm(date_ranges, desc="Downloading MLB Data")
    
    for start_dt, end_dt in pbar:
        try:
            # 1. pybaseball로 데이터 다운로드 (작은 청크)
            # verbose=False로 설정하여 불필요한 로그 숨김
            df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
            
            if df is not None and not df.empty:
                # 2. 데이터 정제 (필요한 경우)
                # 날짜 컬럼을 문자열로 변환 (SQLite 호환성)
                if 'game_date' in df.columns:
                    df['game_date'] = df['game_date'].astype(str)
                
                # 3. SQLite에 저장 (Append 모드)
                df.to_sql('statcast', conn, if_exists='append', index=False)
                
                rows = len(df)
                total_rows += rows
                pbar.set_postfix({'Latest': start_dt, 'Total Rows': total_rows})
                
            # MLB 서버 차단 방지를 위한 짧은 휴식
            time.sleep(1.5)
            
        except Exception as e:
            # 특정 구간 실패해도 멈추지 않고 로그 남기고 계속 진행
            print(f"\n⚠️ 오류 발생 ({start_dt} ~ {end_dt}): {e}")
            continue

    # 4. 인덱스 생성 (데이터 적재 후 한 번만 실행)
    print("\n⚙️ 검색 속도 최적화(Indexing) 중...")
    cursor = conn.cursor()
    # 투수, 타자, 날짜, 구종에 인덱스를 걸어 조회 속도 향상
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pitcher ON statcast (pitcher)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_batter ON statcast (batter)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON statcast (game_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pitch_type ON statcast (pitch_type)")
    conn.commit()
    conn.close()
    
    print(f"\n🎉 적재 완료! 총 {total_rows:,}개의 데이터가 저장되었습니다.")
    print(f"📁 DB 위치: {DB_PATH}")

if __name__ == "__main__":
    download_and_store()