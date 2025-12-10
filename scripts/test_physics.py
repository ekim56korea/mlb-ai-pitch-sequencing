import sys
import os
import pandas as pd

# 우리가 만든 모듈을 불러오기 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.engine.physics import PhysicsEngine

def test_engine():
    print("🔬 물리 엔진 테스트를 시작합니다...")
    
    # 1. 저장해둔 데이터 불러오기
    data_path = os.path.join('data', 'statcast_sample.parquet')
    if not os.path.exists(data_path):
        print("❌ 데이터 파일이 없습니다. download_sample.py를 먼저 실행하세요.")
        return

    df = pd.read_parquet(data_path)
    
    # 2. 첫 번째 투구 데이터 하나 뽑기
    sample_pitch = df.iloc[0]
    player = sample_pitch['player_name']
    pitch_type = sample_pitch['pitch_type']
    
    print(f"⚾️ 투수: {player}, 구종: {pitch_type} 의 궤적을 계산합니다.")
    
    # 3. 엔진 가동
    engine = PhysicsEngine()
    trajectory = engine.calculate_trajectory(sample_pitch)
    
    # 4. 결과 확인
    if len(trajectory) > 0:
        print(f"✅ 궤적 계산 성공!")
        print(f"   - 총 계산된 포인트(프레임) 수: {len(trajectory)}개")
        print(f"   - 릴리스 포인트 (x, y, z): {trajectory[0]}")
        print(f"   - 홈플레이트 도달 (x, y, z): {trajectory[-1]}")
        
        # 검증 (실제 데이터와 비교)
        real_plate_x = sample_pitch['plate_x']
        calc_plate_x = trajectory[-1][0]
        print(f"   - [검증] 실제 plate_x: {real_plate_x:.2f} vs 계산된 x: {calc_plate_x:.2f}")
    else:
        print("❌ 궤적 계산 실패 (데이터 부족 또는 오류)")

if __name__ == "__main__":
    test_engine()