import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

def plot_metric_distribution(score, metric_name="Stuff+"):
    """
    [Data Storytelling] 리그 평균(100, SD=10) 정규분포 곡선 위에
    현재 투구의 위치를 시각화하여 '상대적 위력'을 직관적으로 전달함.
    """
    # 1. 정규분포 데이터 생성 (평균 100, 표준편차 10 가정)
    mean = 100
    std_dev = 10
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 200)
    y = norm.pdf(x, mean, std_dev)

    # 2. 차트 생성
    fig = go.Figure()

    # 분포 곡선 (Bell Curve)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', 
        line=dict(color='gray', width=1), 
        fill='tozeroy', 
        fillcolor='rgba(200, 200, 200, 0.2)',
        name='League Distribution',
        hoverinfo='skip'
    ))

    # 3. 현재 점수 위치 표시 (Vertical Line & Marker)
    # 점수에 따른 색상 (히트맵 컬러)
    if score >= 110: color = '#d73027' # Elite (Red)
    elif score >= 100: color = '#fdae61' # Above Avg (Orange)
    elif score >= 90: color = '#74add1' # Below Avg (Blue)
    else: color = '#4575b4' # Poor (Dark Blue)

    # 수직선
    fig.add_shape(
        type="line",
        x0=score, y0=0, x1=score, y1=max(y)*1.1,
        line=dict(color=color, width=3, dash="dot")
    )

    # 마커 및 텍스트
    percentile = norm.cdf(score, mean, std_dev) * 100
    
    fig.add_trace(go.Scatter(
        x=[score], y=[max(y)*0.5],
        mode='markers+text',
        marker=dict(color=color, size=15, symbol='diamond'),
        text=[f"<b>{score:.0f}</b><br>(Top {100-percentile:.1f}%)"],
        textposition="top center",
        name='Your Pitch',
        textfont=dict(size=14, color=color)
    ))

    # 4. 레이아웃 다듬기
    fig.update_layout(
        title=f"League Context: {metric_name}",
        xaxis=dict(title=f"{metric_name} Score", range=[60, 140], showgrid=False),
        yaxis=dict(showgrid=False, showticklabels=False, visible=False),
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white") # 다크모드 대응
    )

    return fig