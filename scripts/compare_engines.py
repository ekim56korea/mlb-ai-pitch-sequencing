import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.engine.physics import PhysicsEngine as V1Engine
from api.engine.physics_v2 import AdvancedPhysicsEngine as V2Engine

def compare():
    print("🔬 물리 엔진 세대교체 테스트 (V1 vs V2)...")
    
    # 가상의 커브볼 데이터 생성 (커브볼이 물리 효과가 가장 큼)
    # y축 속도(vy0)는 투수->포수이므로 음수입니다. (-132 ft/s ~= 90mph)
    mock_pitch = pd.Series({
        'pitch_type': 'CB', 
        'release_speed': 80.0,
        'release_spin_rate': 2800,
        'release_pos_x': -1.5,
        'release_pos_z': 6.0,
        'vx0': 2.0,
        'vy0': -117.0, # 약 80mph
        'vz0': -4.0,
        'ax': 0, 'ay': 20, 'az': -40 # V1용 가속도 (대충 설정)
    })
    
    # 1. 구형 엔진 (V1) 실행
    v1 = V1Engine()
    traj_v1 = v1.calculate_trajectory(mock_pitch)
    
    # 2. 신형 엔진 (V2) 실행
    v2 = V2Engine()
    traj_v2 = v2.calculate_trajectory(mock_pitch)
    
    print(f"✅ V1 포인트 수: {len(traj_v1)}")
    print(f"✅ V2 포인트 수: {len(traj_v2)}")
    
    if len(traj_v2) > 0:
        final_v1 = traj_v1[-1]
        final_v2 = traj_v2[-1]
        print(f"\n[홈플레이트 도달 위치 비교]")
        print(f"V1 (단순모델): x={final_v1[0]:.2f}, z={final_v1[2]:.2f}")
        print(f"V2 (공기역학): x={final_v2[0]:.2f}, z={final_v2[2]:.2f}")
        
        diff_x = abs(final_v1[0] - final_v2[0]) * 12 # 인치 변환
        diff_z = abs(final_v1[2] - final_v2[2]) * 12 # 인치 변환
        print(f"👉 차이: 가로 {diff_x:.1f}인치, 세로 {diff_z:.1f}인치")
        print("   (이 차이가 바로 공기 저항과 스핀의 효과입니다!)")

if __name__ == "__main__":
    compare()