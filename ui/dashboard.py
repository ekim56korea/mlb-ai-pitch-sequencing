import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# API 서버 주소 (FastAPI가 실행 중인 주소)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pitch Commander Pro", layout="wide")

st.title("⚾️ Pitch Commander Pro: Tactical Dashboard")
st.markdown("### MLB Real-time Decision Support System")

# 화면을 좌우 2단으로 나눔
col_control, col_display = st.columns([1, 2])

with col_control:
    st.header("1. 타자 분석 (Batter Intel)")
    
    # 사용자 입력: 타자 성향 시뮬레이션
    swing_rate = st.slider("Swing Rate (스윙률)", 0.0, 1.0, 0.45)
    whiff_rate = st.slider("Whiff Rate (헛스윙률)", 0.0, 1.0, 0.25)
    chase_rate = st.slider("Chase Rate (유인구 추격률)", 0.0, 1.0, 0.30)
    
    if st.button("🔍 타자 성향 분석 요청"):
        try:
            # API에 분석 요청 보내기
            payload = {
                "swing_rate": swing_rate,
                "whiff_rate": whiff_rate,
                "chase_rate": chase_rate
            }
            response = requests.post(f"{API_URL}/analyze/batter", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"분석 완료: {result['batter_type']}")
                st.info(f"💡 공략 팁: {result['strategy']}")
            else:
                st.error("분석 실패: 서버 오류")
                
        except Exception as e:
            st.error(f"서버 연결 실패: {e}")
            st.warning("⚠️ 백엔드 서버(uvicorn)가 켜져 있는지 확인하세요!")

    st.markdown("---")
    st.header("2. 투구 시뮬레이션 (Physics)")
    pitch_type = st.selectbox("구종 선택", ["FF (Four-Seam)", "SL (Slider)", "CB (Curve)"])
    velo = st.slider("구속 (mph)", 70.0, 105.0, 93.0)

    st.markdown("---")
    st.header("1.5. 전략 수립 (Strategy)")
    
    # 볼카운트 선택
    ball_count = st.selectbox("볼 카운트 (Ball-Strike)", ["0-0", "1-0", "0-1", "0-2", "3-0", "3-2 (Full)"])
    
    # 타자 분석 결과가 있을 때만 활성화 (세션 스테이트 등을 쓰면 좋지만 여기선 간단히)
    # 편의상 사용자가 직접 Cluster ID를 입력하게 하거나, 위에서 분석된 결과를 기억해야 함.
    # 여기서는 테스트를 위해 수동 입력 허용
    current_cluster = st.number_input("현재 타자 클러스터 ID (0~4)", min_value=0, max_value=4, value=0)

    if st.button("🧠 AI 투구 추천 받기"):
        try:
            # 쿼리 파라미터로 전송
            response = requests.post(f"{API_URL}/recommend/strategy", params={"cluster_id": current_cluster, "ball_count": ball_count})
            
            if response.status_code == 200:
                rec = response.json()
                st.success(f"🎯 추천 구종: {rec['recommended_pitch']}")
                st.caption(f"사유: {rec['reasoning']}")
                
                # 데이터 시각화 (xRV 비교)
                st.bar_chart(rec['mix_strategy'])
                st.info("그래프가 낮을수록 투수에게 유리한 구종입니다.")
            else:
                st.error("추천 실패")
        except Exception as e:
            st.error(f"연결 오류: {e}")
    
    if st.button("🚀 궤적 시뮬레이션"):
        try:
            # API에 궤적 계산 요청
            payload = {
                "pitch_type": pitch_type.split()[0],
                "release_speed": velo,
                "release_spin_rate": 2200 # 임시 기본값
            }
            response = requests.post(f"{API_URL}/simulate/trajectory", json=payload)
            
            if response.status_code == 200:
                traj_data = response.json()
                
                # Plotly로 3D 궤적 그리기
                fig = go.Figure(data=[go.Scatter3d(
                    x=traj_data['x'],
                    y=traj_data['y'],
                    z=traj_data['z'],
                    mode='lines',
                    line=dict(color='red', width=5),
                    name='Pitch Trajectory'
                )])
                
                # 홈플레이트 및 마운드 표시 (참조선)
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title='X (좌우)', range=[-3, 3]),
                        yaxis=dict(title='Y (거리)', range=[0, 60]),
                        zaxis=dict(title='Z (높이)', range=[0, 8]),
                    ),
                    title=f"3D Trajectory Simulation: {pitch_type} @ {velo}mph",
                    height=600
                )
                
                with col_display:
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric(label="최종 위치 (Plate X)", value=f"{traj_data['final_x']:.2f} ft")
                    st.metric(label="최종 높이 (Plate Z)", value=f"{traj_data['final_z']:.2f} ft")
                    
            else:
                st.error("시뮬레이션 실패")
        except Exception as e:
            st.error(f"연결 오류: {e}")

with col_display:
    st.info("👈 왼쪽 패널에서 타자 정보를 입력하거나 투구 조건을 설정하세요.")
    # 기본 이미지나 설명 텍스트