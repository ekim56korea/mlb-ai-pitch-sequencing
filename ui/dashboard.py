import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta

# [Path Setup]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try: from viz_utils import plot_metric_distribution
except ImportError:
    def plot_metric_distribution(score, name): return go.Figure()

# API 서버 주소
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pitch Commander Pro v7.0", layout="wide", page_icon="⚾")
st.title("⚾️ Pitch Commander Pro: Zero-Cost Edition (v7.0)")

# ==================== [Helper Functions: Caching & UX] ====================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_matchup_data(p_name, b_name, start_dt, end_dt):
    """
    [Phase 5] 데이터 로딩 캐싱
    - 한 번 로드한 매치업은 1시간(3600초) 동안 캐시에서 즉시 반환
    - 네트워크 비용 0, 대기 시간 0.01초
    """
    params = {
        "pitcher_name": p_name, "batter_name": b_name,
        "start_dt": start_dt, "end_dt": end_dt
    }
    try:
        res = requests.post(f"{API_URL}/load/matchup", params=params)
        if res.status_code == 200: return res.json()
        return None
    except: return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_volumetric_data(b_name, start_dt, end_dt):
    """[Phase 5] 볼류메트릭 핫존 캐싱 (10분)"""
    try:
        res = requests.post(f"{API_URL}/analyze/volumetric", 
                            params={"batter_name": b_name, "start_dt": start_dt, "end_dt": end_dt})
        return res.json()
    except: return {"status": "error", "data": []}

# 시뮬레이션은 변수가 너무 많아 캐싱보다는 UX(Status)에 집중
def run_simulation(payload):
    try:
        traj = requests.post(f"{API_URL}/simulate/trajectory", json=payload).json()
        metrics = requests.post(f"{API_URL}/analyze/metrics", json=payload).json()
        return traj, metrics
    except Exception as e:
        return None, None

# ==================== [SIDEBAR] ====================
with st.sidebar:
    st.header("1. Matchup Setup")
    col_p, col_b = st.columns(2)
    with col_p: p_name = st.text_input("Pitcher", "Cole Gerrit")
    with col_b: b_name = st.text_input("Batter", "Ohtani Shohei")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1: start_date = st.date_input("Start", datetime.now() - timedelta(days=365))
    with col_d2: end_date = st.date_input("End", datetime.now())

    if st.button("📥 Load Data", type="primary"):
        # [Phase 5 UX] Status Container
        with st.status("🔍 Scouting Players...", expanded=True) as status:
            st.write("Connecting to Local Warehouse...")
            data = fetch_matchup_data(p_name, b_name, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            if data:
                st.session_state['matchup'] = data
                st.write("Analyzing Pitch Arsenal...")
                status.update(label="Data Loaded Successfully!", state="complete", expanded=False)
                st.toast(f"Matchup Loaded: {p_name} vs {b_name}", icon="✅")
                
                # 유사 투수 정보 표시
                if 'archetype' in data['pitcher'] and data['pitcher']['archetype']:
                    arch = data['pitcher']['archetype']
                    st.info(f"🧬 **Archetype**: {arch['name']} ({arch['type']}) - {arch['similarity']}% Match")
            else:
                status.update(label="Player Not Found", state="error")
                st.error("Failed to load player data. Check names.")

    st.markdown("---")
    st.header("2. Environment")
    temp = st.slider("Temp (F)", 30, 100, 70)
    elev = st.slider("Elevation (ft)", 0, 5200, 0)
    humid = st.slider("Humidity (%)", 0, 100, 50)
    
    st.markdown("#### 🌬️ Wind Field")
    wind_spd = st.slider("Speed (mph)", 0, 30, 0)
    wind_dir = st.slider("Direction (deg)", 0, 360, 0)

# ==================== [CONTEXT] ====================
st.subheader("🏟️ Game Context (Leverage & Strategy)")
with st.container():
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        inning = st.number_input("Inning", 1, 12, 1)
        outs = st.selectbox("Outs", [0, 1, 2])
    with c2:
        balls = st.selectbox("Balls", [0, 1, 2, 3])
        strikes = st.selectbox("Strikes", [0, 1, 2])
    with c3:
        score_diff = st.number_input("Score Diff", -10, 10, 0, help="+: Leading, -: Trailing")
        r1 = st.checkbox("1B"); r2 = st.checkbox("2B"); r3 = st.checkbox("3B")
    with c4:
        st.caption("Context drives Leverage Index (LI) & AI Strategy.")

# ==================== [MAIN] ====================
c_left, c_right = st.columns([1, 2])

# --- [LEFT] AI Strategy ---
with c_left:
    st.markdown("### 🤖 Tactical Command (Deep RL)")
    arsenal_keys = ["FF", "SL", "CH", "CB", "SI"]
    if 'matchup' in st.session_state:
        try: 
            keys = list(st.session_state['matchup']['pitcher']['arsenal'].keys())
            if keys: arsenal_keys = keys
        except: pass

    if st.button("🧠 AI 전략 분석"):
        with st.spinner("Thinking..."):
            payload = {
                "context": {"inning": inning, "balls": balls, "strikes": strikes, "outs": outs,
                            "runner_on_1b": r1, "runner_on_2b": r2, "runner_on_3b": r3, "score_diff": score_diff},
                "arsenal": arsenal_keys
            }
            try:
                res = requests.post(f"{API_URL}/recommend/context", json=payload["context"], params={"arsenal": arsenal_keys})
                st.session_state['recommendation'] = res.json()
            except: st.error("AI Engine Offline")

    if 'recommendation' in st.session_state:
        rec = st.session_state['recommendation']
        
        # Leverage Index Badge
        li = rec.get('leverage_index', 1.0)
        li_color = "red" if li > 2.0 else "green"
        st.markdown(f"**Strategy:** {rec['strategy_name']} <span style='background-color:{li_color}; color:white; padding:2px 6px; border-radius:4px; font-size:12px'>LI {li}</span>", unsafe_allow_html=True)
        
        st.write(f"💡 {rec['reasoning']}")
        
        st.markdown("---")
        
        # Swing Probability Gauge
        if 'swing_prob' in rec:
            sp = rec['swing_prob']
            st.markdown(f"**Swing Prob: {sp}%**")
            st.progress(sp / 100.0)
        
        # Batter Anticipation
        if 'guess_probs' in rec:
            st.markdown("**Batter Anticipation**")
            probs = rec['guess_probs']
            df_probs = pd.DataFrame(list(probs.items()), columns=['Pitch', 'Probability'])
            colors = ['#ff4b4b' if x >= 50 else '#4575b4' for x in df_probs['Probability']]
            fig_guess = go.Figure(go.Bar(
                x=df_probs['Probability'], y=df_probs['Pitch'], orientation='h',
                marker_color=colors, text=df_probs['Probability'].apply(lambda x: f"{x}%"), textposition='auto'
            ))
            fig_guess.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=120, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_guess, use_container_width=True)
            
        # Sequence
        st.markdown("---")
        c_s1, c_s2, c_s3 = st.columns([1, 0.2, 1])
        with c_s1: st.info(f"**① Setup**\n\n{rec['recommended_pitch']}\n{rec['location_desc']}")
        with c_s2: st.markdown("## →")
        with c_s3: 
            if 'next_pitch' in rec: st.warning(f"**② Finish**\n\n{rec['next_pitch']['pitch']}\n{rec['next_pitch']['location']}")
        
        ai_tx, ai_tz = rec['target_x'], rec['target_z']
    else:
        ai_tx, ai_tz = 0.0, 2.5

# --- [RIGHT] Simulation ---
with c_right:
    st.markdown("### 🧪 Hyper-Simulation")
    
    sq1, sq2 = st.columns(2)
    with sq1: pitch_count = st.number_input("Pitch Count", 0, 150, 15)
    with sq2: use_prev = st.checkbox("Prev Pitch (Tunneling)")
    prev_data = None
    if use_prev:
        pp1, pp2, pp3 = st.columns(3)
        pt = pp1.selectbox("Prev Type", ["FF", "SL"], index=0)
        px = pp2.number_input("Prev X", -2.0, 2.0, 0.0)
        pz = pp3.number_input("Prev Z", 0.0, 5.0, 2.5)
        prev_data = {"pitch_type": pt, "release_speed": 95, "pfx_x": 0, "pfx_z": 10, "plate_x": px, "plate_z": pz}

    st.markdown("---")
    
    sc1, sc2 = st.columns(2)
    with sc1:
        u_pitch = st.selectbox("Pitch", arsenal_keys)
        defaults = {'release_speed': 93.0, 'release_spin_rate': 2200, 'pfx_x': -5.0, 'pfx_z': 10.0}
        if 'matchup' in st.session_state:
            try: defaults = st.session_state['matchup']['pitcher']['arsenal'].get(u_pitch, defaults)
            except: pass
        u_velo = st.slider("Velo", 70.0, 105.0, float(defaults['release_speed']))
        u_spin = st.slider("Spin", 1000, 3500, int(defaults['release_spin_rate']))
    with sc2:
        u_px = st.slider("Horz", -25.0, 25.0, float(defaults['pfx_x']))
        u_pz = st.slider("Vert", -25.0, 25.0, float(defaults['pfx_z']))
        
    with st.expander("🧬 Advanced Physics (SSW & Seam)"):
        ap1, ap2 = st.columns(2)
        with ap1: spin_eff = st.slider("Spin Eff %", 0, 100, 100)
        with ap2: seam_lat = st.slider("Seam Lat °", -90, 90, 0)
        
    # Physics Lab Output Placeholder
    phys_container = st.empty()

    st.markdown("#### 🎯 Target (Auto-Aim)")
    l1, l2 = st.columns(2)
    with l1: target_x = st.slider("Plate X", -2.0, 2.0, 0.0)
    with l2: target_z = st.slider("Plate Z", 0.0, 5.0, 2.5)

    if st.button("🚀 Score This Pitch", type="primary", use_container_width=True):
        ctx = {"inning": inning, "balls": balls, "strikes": strikes, "outs": outs, 
               "runner_on_1b": r1, "runner_on_2b": r2, "runner_on_3b": r3, "score_diff": score_diff}
        payload = {
            "pitch_type": u_pitch, "release_speed": u_velo, "release_spin_rate": u_spin,
            "pfx_x": u_px, "pfx_z": u_pz, "extension": 6.0,
            "spin_efficiency": spin_eff, "seam_lat": seam_lat,
            "plate_x": target_x, "plate_z": target_z, "pitch_count": pitch_count, "prev_pitch": prev_data,
            "env": {"temperature": temp, "elevation": elev, "humidity": humid, "wind_speed": wind_spd, "wind_direction": wind_dir},
            "context": ctx
        }
        
        # [Phase 5 UX] Status for Simulation
        with st.status("Computing Physics & Metrics...", expanded=True) as status:
            st.write("Solving ODEs (Physics Engine)...")
            traj, metrics = run_simulation(payload)
            
            if traj and metrics:
                # Ghost Calculation
                ghost_traj = None
                if prev_data:
                    st.write("Analyzing Tunneling Effect...")
                    g_payload = payload.copy()
                    g_payload.update({"pitch_type": prev_data['pitch_type'], "plate_x": prev_data['plate_x'], "plate_z": prev_data['plate_z']})
                    ghost_traj, _ = run_simulation(g_payload)
                    
                status.update(label="Simulation Complete!", state="complete", expanded=False)
                
                # --- Scoreboard ---
                st.markdown("#### 📊 Pitching+ Scoreboard")
                sample_size = 0
                if 'matchup' in st.session_state:
                    try: sample_size = int(st.session_state['matchup']['pitcher']['arsenal'][u_pitch].get('count', 0))
                    except: pass
                
                def calc_shrink(s, n, c=50): return (n/(n+c)*s) + ((1-(n/(n+c)))*100) if n>0 else 100
                s_raw = metrics.get('stuff_plus', 0)
                s_adj = calc_shrink(s_raw, sample_size)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Stuff+", f"{s_raw:.0f}", delta=f"Adj: {s_adj:.0f}", delta_color="off")
                m2.metric("Location+", f"{metrics.get('location_plus',0):.0f}")
                m3.metric("Pitching+", f"{metrics.get('pitching_plus',0):.0f}")
                m4.metric("Confidence", "High" if sample_size>100 else "Low", delta=f"N={sample_size}")
                
                st.plotly_chart(plot_metric_distribution(s_raw, "Stuff+"), use_container_width=True)

                # Physics Lab Display
                if 'physics_est' in traj:
                    est = traj['physics_est']; cont = traj['contact_est']
                    with phys_container.container():
                        st.caption("📐 Physics Lab Result")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Gyro", f"{est['gyro_degree']}°", help="Efficiency: "+str(est['efficiency'])+"%")
                        c2.metric("Exit Velo", f"{cont['exit_velocity']} mph")
                        c3.metric("Distance", f"{cont['est_distance']} ft")

                # --- Tabs: 3D Viz vs Volumetric ---
                tab1, tab2 = st.tabs(["🏟️ 3D Trajectory", "🧊 Volumetric Lab"])
                
                with tab1:
                    fig = go.Figure()
                    if ghost_traj:
                        fig.add_trace(go.Scatter3d(x=ghost_traj['x'], y=ghost_traj['y'], z=ghost_traj['z'], mode='lines', line=dict(color='gray', width=4, dash='dash'), opacity=0.5, name='Ghost'))
                    fig.add_trace(go.Scatter3d(x=traj['x'], y=traj['y'], z=traj['z'], mode='lines', line=dict(color='#ff4b4b', width=6), name='Real'))
                    fig.add_trace(go.Scatter3d(x=[target_x], y=[1.417], z=[target_z], mode='markers', marker=dict(size=8, color='cyan', symbol='x'), name='Target'))
                    
                    # Zone
                    zone_x = [-0.71, 0.71, 0.71, -0.71, -0.71]; y_plane = [1.417]*5
                    fig.add_trace(go.Scatter3d(x=zone_x, y=y_plane, z=[1.5]*5, mode='lines', line=dict(color='white'), showlegend=False))
                    fig.add_trace(go.Scatter3d(x=zone_x, y=y_plane, z=[3.5]*5, mode='lines', line=dict(color='white'), showlegend=False))
                    for i in range(4): fig.add_trace(go.Scatter3d(x=[zone_x[i], zone_x[i]], y=[1.417,1.417], z=[1.5,3.5], mode='lines', line=dict(color='white'), showlegend=False))
                    
                    fig.update_layout(scene=dict(xaxis=dict(range=[-3,3], showgrid=False, backgroundcolor="black"), yaxis=dict(range=[0,60.5], showgrid=False, backgroundcolor="black"), zaxis=dict(range=[0,6], showgrid=False, backgroundcolor="black"), aspectratio=dict(x=1,y=3,z=1)), margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="black", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    # [Phase 5 Cache] Volumetric Data
                    v_res = fetch_volumetric_data(b_name, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    if v_res['status'] == 'success' and v_res['data']:
                        df_vol = pd.DataFrame(v_res['data'])
                        color_map = {'Weakness (Strike/Whiff)': '#ff4b4b', 'Strength (Contact)': '#4575b4', 'Neutral (Ball)': 'gray'}
                        fig_vol = go.Figure(data=[go.Scatter3d(x=df_vol['horz_break'], y=df_vol['velocity'], z=df_vol['vert_break'], mode='markers', marker=dict(size=4, color=df_vol['result'].map(color_map), opacity=0.7), hovertext=df_vol['pitch_type'])])
                        fig_vol.update_layout(scene=dict(xaxis_title='Horz', yaxis_title='Velo', zaxis_title='Vert', xaxis=dict(backgroundcolor="black"), yaxis=dict(backgroundcolor="black"), zaxis=dict(backgroundcolor="black")), margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor="black", height=400)
                        st.plotly_chart(fig_vol, use_container_width=True)
                    else: st.info("No Volumetric Data Available")

            else: st.error("Simulation Failed")