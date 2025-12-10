import sys
import os
import pandas as pd

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.engine.ai_models import BatterClustering

def run_training():
    # 1. 데이터 로드
    data_path = os.path.join('data', 'statcast_sample.parquet')
    if not os.path.exists(data_path):
        print("❌ 데이터 파일이 없습니다.")
        return

    print("📂 데이터를 불러오는 중...")
    df = pd.read_parquet(data_path)

    # 2. AI 모델 초기화 및 학습
    clustering_engine = BatterClustering(n_clusters=5)
    
    # 학습 결과(타자별 클러스터) 받기
    result_df = clustering_engine.train(df)

    # 3. 결과 분석 출력
    print("\n[🏆 타자 유형별 분석 결과]")
    print(result_df.groupby('cluster').mean().round(3))
    
    print("\n---> 해석 가이드:")
    print("swing_rate: 높을수록 공격적")
    print("whiff_rate: 높을수록 헛스윙 많음 (정교함 부족)")
    print("chase_rate: 높을수록 나쁜 공에 잘 속음")

    # 4. 모델 저장
    model_dir = os.path.join('api', 'engine')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'batter_cluster_model.pkl')
    clustering_engine.save_model(save_path)

if __name__ == "__main__":
    run_training()