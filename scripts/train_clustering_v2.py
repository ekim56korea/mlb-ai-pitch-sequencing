import sys
import os
import sqlite3
import pandas as pd

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api.engine.ai_models import BatterClustering

DB_PATH = os.path.join("data", "mlb_statcast.db")

def train_v2():
    print("🚀 AI 모델 V2 학습을 시작합니다 (Source: SQLite DB)")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. 학습 데이터 로드 (2024년 데이터만 사용)
    # 메모리 절약을 위해 필요한 컬럼만 가져옵니다.
    print("📥 2024년 시즌 데이터를 불러오는 중... (약간의 시간이 소요됩니다)")
    query = """
    SELECT batter, pitch_type, description, zone
    FROM statcast
    WHERE game_date >= '2024-01-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"📦 학습 데이터 로드 완료: {len(df):,} 건")
    
    # 2. 모델 학습
    clustering_engine = BatterClustering(n_clusters=5)
    result_df = clustering_engine.train(df)

    # 3. 결과 요약
    print("\n[🏆 V2 타자 유형별 분석 결과 (Big Data 기반)]")
    summary = result_df.groupby('cluster').mean().round(3)
    print(summary)
    
    # 4. 모델 덮어쓰기 (API가 바로 사용할 수 있도록)
    model_dir = os.path.join('api', 'engine')
    save_path = os.path.join(model_dir, 'batter_cluster_model.pkl')
    clustering_engine.save_model(save_path)
    print("✅ V2 모델이 적용되었습니다. API 서버를 재시작하면 반영됩니다.")

if __name__ == "__main__":
    train_v2()