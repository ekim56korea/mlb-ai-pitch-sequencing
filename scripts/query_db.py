import sqlite3
import pandas as pd
import time
import os

DB_PATH = os.path.join("data", "mlb_statcast.db")

def query_player_stats(player_name_fragment):
    conn = sqlite3.connect(DB_PATH)
    
    print(f"🔍 '{player_name_fragment}' 선수의 데이터를 300만 건 중에서 검색합니다...")
    start_time = time.time()
    
    # SQL 쿼리: 이름 검색 -> 구종별 평균 구속 및 회전수 계산
    # (3년치 데이터를 한방에 집계)
    query = f"""
    SELECT 
        player_name,
        pitch_type,
        COUNT(*) as pitch_count,
        ROUND(AVG(release_speed), 1) as avg_speed,
        ROUND(AVG(release_spin_rate), 0) as avg_spin
    FROM statcast
    WHERE player_name LIKE '%{player_name_fragment}%'
    GROUP BY player_name, pitch_type
    ORDER BY pitch_count DESC
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        duration = time.time() - start_time
        
        if df.empty:
            print("❌ 선수를 찾을 수 없습니다.")
        else:
            print(f"✅ 검색 완료! (소요 시간: {duration:.4f}초)")
            print("\n[📊 구종 분석 결과]")
            print(df.to_string(index=False))
            
    except Exception as e:
        print(f"⚠️ 쿼리 오류: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # 예시: 오타니 쇼헤이(Ohtani) 검색
    query_player_stats("Ohtani, Shohei")