import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pitch Commander Pro v4.0", layout="wide")
st.title("⚾️ Pitch Commander Pro: Tactical Command (v4.0)")

# ==================== [SIDEBAR: 데이터 및 환경] ====================
with st.sidebar:
    st.header("1. Matchup Setup")
    col_p, col_b = st.columns(2)
    with col_p: p_name = st.text_input("Pitcher", "Cole Gerrit")
    with col_b: b_name = st.text_input("Batter", "Ohtani Shohei")
    
    if st.button("📥 Load Data", type="primary"):
        with st.spinner("Loading Statcast Data..."):
            try:
                res = requests.post(f"{API_URL}/load/matchup", 
                                    params={"pitcher_name": p_name, "batter_name": b_name})
                if res.status_code == 200:
                    st.session_state['matchup'] = res.json()
                    st.success("Loaded!")
                else: st.error("Failed to load")
            except: st.error("Connection Error")

    st.markdown("---")
    st.header("2. Environment")
    temp = st.slider("Temp (F)", 30, 100, 70)
    elev = st.slider("Elevation (ft)", 0, 5200, 0)
    humid = st.slider("Humidity (%)", 0, 100, 50)

# ==================== [TOP: 경기 상황 입력] ====================
st.subheader("🏟️ Game Context (Situation)")
with st.container():
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        inning = st.number_input("Inning", 1, 12, 1)
        outs = st.selectbox("Outs", [0, 1, 2])
    with c2:
        balls = st.selectbox("Balls", [0, 1, 2, 3])
        strikes = st.selectbox("Strikes", [0, 1, 2])
    with c3:
        st.markdown("**Runners**")
        r1 = st.checkbox("1st Base")
        r2 = st.checkbox("2nd Base")
        r3 = st.checkbox("3rd Base")
    with c4:
        st.info("👈 상황을 설정하면 AI가 최적의 전략을 계산합니다.")

# ==================== [MAIN: 추천 vs 시뮬레이션] ====================
c_left, c_right = st.columns([1, 2])

# --- [LEFT] AI 전략 추천 ---
with c_left:
    st.markdown("### 🤖 AI Strategy")
    
    arsenal_keys = ["FF", "SL", "CH", "CB", "SI"]
    if 'matchup' in st.session_state:
        arsenal_keys = list(st.session_state['matchup']['pitcher']['arsenal'].keys())

    # 추천 요청
    rec_btn = st.button("🧠 전략 분석 (Get Recommendation)")
    
    if rec_btn:
        payload = {
            "context": {"inning": inning, "balls": balls, "strikes": strikes, "outs": outs,
                        "runner_on_1b": r1, "runner_on_2b": r2, "runner_on_3b": r3},
            "arsenal": arsenal_keys
        }
        try:
            rec_res = requests.post(f"{API_URL}/recommend/context", json=payload["context"], params={"arsenal": arsenal_keys}).json()
            st.session_state['recommendation'] = rec_res
        except Exception as e: st.error(f"Error: {e}")

    if 'recommendation' in st.session_state:
        rec = st.session_state['recommendation']
        st.success(f"**{rec['strategy_name']}**")
        st.markdown(f"### 👉 Recommend: **{rec['recommended_pitch']}**")
        st.caption(f"Target: {rec['location_desc']}")
        st.write(f"💡 {rec['reasoning']}")
        
        # AI 타겟 좌표 저장 (시각화용)
        ai_target_x = rec['target_x']
        ai_target_z = rec['target_z']
    else:
        ai_target_x, ai_target_z = 0.0, 2.5 # Default

# --- [RIGHT] 사용자 시뮬레이션 & 점수판 ---
with c_right:
    st.markdown("### 🧪 Simulation & Pitching+ Scoreboard")
    
    # 1. 사용자 투구 설정 (User Simulation)
    sc1, sc2 = st.columns(2)
    with sc1:
        u_pitch = st.selectbox("구종 선택", arsenal_keys, index=0)
        # 로드된 데이터가 있으면 평균값 가져오기
        defaults = {'release_speed': 93.0, 'release_spin_rate': 2200, 'pfx_x': -5.0, 'pfx_z': 10.0}
        if 'matchup' in st.session_state:
            defaults = st.session_state['matchup']['pitcher']['arsenal'].get(u_pitch, defaults)
            
        u_velo = st.slider("구속 (mph)", 70.0, 105.0, float(defaults['release_speed']))
        u_spin = st.slider("회전수 (rpm)", 1000, 3500, int(defaults['release_spin_rate']))
        
    with sc2:
        u_pfx_x = st.slider("Horz Break (in)", -20.0, 20.0, float(defaults['pfx_x']))
        u_pfx_z = st.slider("Vert Break (in)", -20.0, 20.0, float(defaults['pfx_z']))
        
    run_sim = st.button("🚀 시뮬레이션 실행 (Score This Pitch)", type="primary", use_container_width=True)

    # 2. 결과 처리 및 시각화
    if run_sim:
        payload = {
            "pitch_type": u_pitch, "release_speed": u_velo, "release_spin_rate": u_spin,
            "pfx_x": u_pfx_x, "pfx_z": u_pfx_z, "extension": 6.0,
            "env": {"temperature": temp, "elevation": elev, "humidity": humid}
        }
        
        try:
            # Physics & Metrics API 호출
            traj = requests.post(f"{API_URL}/simulate/trajectory", json=payload).json()
            metrics = requests.post(f"{API_URL}/analyze/metrics", json=payload).json()
            
            # --- [Scoreboard] ---
            st.markdown("#### 📊 Pitching+ Scoreboard")
            m1, m2, m3, m4 = st.columns(4)
            
            s_val = metrics['stuff_plus']
            m1.metric("Stuff+", f"{s_val}", delta=f"{s_val-100:.1f}")
            m2.metric("xRV", f"{metrics['xRV']}", delta_color="inverse")
            m3.metric("VAA", f"{traj['approach_angle_v']:.1f}°")
            m4.metric("HAA", f"{traj['approach_angle_h']:.1f}°")
            
            # --- [3D Visualization] ---
            fig = go.Figure()
            
            # 1) 사용자 투구 궤적
            fig.add_trace(go.Scatter3d(
                x=traj['x'], y=traj['y'], z=traj['z'],
                mode='lines', line=dict(color='#ff4b4b', width=6),
                name=f'User: {u_pitch}'
            ))
            
            # 2) AI 추천 타겟 (투명한 구체로 표시)
            if 'recommendation' in st.session_state:
                fig.add_trace(go.Scatter3d(
                    x=[ai_target_x], y=[1.417], z=[ai_target_z],
                    mode='markers', marker=dict(size=10, color='green', opacity=0.8),
                    name=f"AI Target ({st.session_state['recommendation']['recommended_pitch']})"
                ))

            # 3) 스트라이크 존 (Wireframe Box)
            zone_x = [-0.71, 0.71, 0.71, -0.71, -0.71]
            zone_z_b = [1.5, 1.5, 1.5, 1.5, 1.5]
            zone_z_t = [3.5, 3.5, 3.5, 3.5, 3.5]
            y_plane = [1.417] * 5
            
            fig.add_trace(go.Scatter3d(x=zone_x, y=y_plane, z=zone_z_b, mode='lines', line=dict(color='white'), showlegend=False))
            fig.add_trace(go.Scatter3d(x=zone_x, y=y_plane, z=zone_z_t, mode='lines', line=dict(color='white'), showlegend=False))
            for i in range(4): # 기둥
                fig.add_trace(go.Scatter3d(x=[zone_x[i], zone_x[i]], y=[1.417, 1.417], z=[1.5, 3.5], mode='lines', line=dict(color='white'), showlegend=False))

            # 4) 홈플레이트
            hp_x = [0, 0.71, 0.71, -0.71, -0.71, 0]
            hp_y = [0, 0.5, 1.417, 1.417, 0.5, 0]
            fig.add_trace(go.Scatter3d(x=hp_x, y=hp_y, z=[0]*6, mode='lines', line=dict(color='white'), name='Home Plate'))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-3, 3], title="", showgrid=False, backgroundcolor="black"),
                    yaxis=dict(range=[0, 60.5], title="", showgrid=False, backgroundcolor="black"),
                    zaxis=dict(range=[0, 6], title="", showgrid=False, backgroundcolor="black"),
                    aspectratio=dict(x=1, y=3, z=1),
                    camera=dict(eye=dict(x=0, y=2.5, z=0.5))
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor="black", height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Simulation Error: {e}")