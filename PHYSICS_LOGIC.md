#### 3. `PHYSICS_LOGIC.md` (물리 엔진 원리)
> "어떻게 공이 휘나요?"에 대한 수학적 답변입니다.

```markdown
# ⚛️ Physics & Simulation Logic

이 프로젝트는 실제 야구공의 물리적 움직임을 시뮬레이션하기 위해 **Kinematics(운동학)** 방정식을 사용합니다.

## 1. 3D Trajectory Calculation (궤적 계산)
투구 궤적은 시간 $t$에 따른 위치 벡터 $P(t) = (x, y, z)$로 정의됩니다. 우리는 MLB 데이터(PITCHf/x)를 역산하여 가속도를 구합니다.

### 기본 운동 방정식
$$x(t) = x_0 + v_{x0}t + \frac{1}{2}a_x t^2$$
$$y(t) = y_0 + v_{y0}t + \frac{1}{2}a_y t^2$$
$$z(t) = z_0 + v_{z0}t + \frac{1}{2}a_z t^2$$

### 가속도 역산 (Reverse Engineering)
MLB 데이터는 `pfx_x` (수평 무브먼트)와 `pfx_z` (수직 무브먼트)를 제공합니다. 이를 가속도로 변환합니다.

* **수평 가속도 ($a_x$):** 마그누스 힘에 의한 횡적인 움직임.
    $$a_x = \frac{2 \times (-pfx\_x)}{t_{flight}^2}$$
* **수직 가속도 ($a_z$):** 중력($g = -32.174 ft/s^2$)과 마그누스 양력(Lift)의 합.
    $$a_z = -g + \frac{2 \times pfx\_z}{t_{flight}^2}$$

## 2. Coordinate System Mapping (좌표계 변환)
MLB의 좌표계와 Three.js의 좌표계는 축이 다릅니다. 이를 보정합니다.

| MLB (Statcast) | Three.js (Project) | 설명 |
| :--- | :--- | :--- |
| **X축** | **-X축** | 포수 시점에서 좌우 (좌우 반전 필요) |
| **Y축** | **Z축** | 투수판에서 홈플레이트까지의 거리 (깊이) |
| **Z축** | **Y축** | 지면에서의 높이 |

## 3. Heatmap Algorithm (히트맵 알고리즘)
스트라이크 존을 $N \times M$ 격자(Grid)로 나누고, 각 투구의 착탄점(`plate_x`, `plate_z`)이 어느 격자에 속하는지 계산합니다.

1.  **Grid Size:** $0.25 ft$ (약 7.6cm) 단위로 분할.
2.  **Density Calculation:** 각 격자에 포함된 투구 수를 카운트.
3.  **Color Mapping:** 최대 밀도 대비 현재 밀도 비율에 따라 Blue(Low) $\rightarrow$ Red(High) 색상 보간(Interpolation) 적용.