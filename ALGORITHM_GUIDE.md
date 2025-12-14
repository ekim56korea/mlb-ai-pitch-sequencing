```markdown
# 🧠 Algorithm & Logic Deep Dive

이 문서는 프로젝트에 적용된 **물리학(Physics), 인공지능(AI), 통계학(Statistics)** 알고리즘의 상세 로직을 설명합니다.

---

## 1. 3D Physics Engine (물리 엔진 로직)

단순한 베지에 곡선(Bezier Curve)이 아닌, **운동학(Kinematics) 방정식**을 역산하여 실제 투구 궤적을 재현했습니다.

### 1.1. 가속도 역산 (Reverse Engineering Acceleration)
MLB Statcast 데이터는 공의 '최종 위치'와 '무브먼트(pfx)'를 제공합니다. 우리는 이를 통해 공이 받는 가속도를 역산출합니다.

* **수평 가속도 ($a_x$):** 마그누스 힘에 의한 횡적 움직임
  $$a_x = \frac{-pfx\_x \times 12}{0.5 \times t_{flight}^2} \quad (\text{ft/s}^2)$$
* **수직 가속도 ($a_z$):** 중력($g$)과 양력(Lift)의 합
  $$a_z = -g + \frac{pfx\_z \times 12}{0.5 \times t_{flight}^2} \quad (\text{ft/s}^2)$$

### 1.2. 궤적 생성 (Trajectory Generation)
계산된 가속도를 이용해 $t=0$부터 $t=flight\_time$까지 0.01초 단위로 위치 벡터 $P(t)$를 계산하여 3D 튜브(TubeGeometry)를 생성합니다.

$$P(t) = P_0 + v_0 t + \frac{1}{2} a t^2$$

---

## 2. 3D Heatmap Algorithm (히트맵 알고리즘)

수천 개의 투구 데이터를 3D 공간 상의 밀도(Density)로 변환하기 위해 **Spatial Binning** 알고리즘을 사용합니다.

1.  **Grid System:** 스트라이크 존 주변을 $0.25ft \times 0.25ft$ 크기의 격자(Grid)로 분할합니다.
2.  **Binning:** 각 투구의 착탄점 $(x, z)$를 격자 인덱스 $(i, j)$로 매핑합니다.
    $$i = \lfloor (x - x_{min}) / grid\_size \rfloor$$
3.  **Normalization:** 각 격자의 투구 수($Count$)를 최대 투구 수($Max$)로 나누어 강도($Intensity$)를 계산합니다.
4.  **Color Interpolation:** 강도에 따라 HSL 색상 공간에서 Blue(0.6) $\to$ Red(0.0)로 보간하여 시각화합니다.

---

## 3. AI Prediction Model (투구 예측 모델)

### 3.1. Model Architecture
-   **Algorithm:** Random Forest Classifier (Ensemble Learning)
-   **Justification:** 투구 패턴은 비선형적이며, 특정 상황(볼카운트 등)에 따라 분기되는 의사결정 나무(Decision Tree) 구조와 가장 유사하기 때문입니다.

### 3.2. Feature Engineering (입력 변수)
모델은 다음 6가지 상황 변수를 입력받아 학습합니다.
1.  **Pitcher Name:** 투수 식별자 (One-hot Encoding or Label Encoding)
2.  **Batter Stand:** 좌타자(L) vs 우타자(R)
3.  **Balls & Strikes:** 볼카운트 심리전 반영
4.  **Outs & Inning:** 경기 후반부/위기 상황 반영

### 3.3. Inference (추론)
`predict_proba()` 메서드를 사용하여 단순한 클래스 분류가 아닌, 각 구종별 **확률(Probability)**을 반환합니다.
> 예: Fastball(60%), Slider(30%), Curve(10%)

---

## 4. Outcome Simulation Logic (결과 시뮬레이션)

사용자가 특정 구종을 선택했을 때, 과거 데이터를 기반으로 **기대 성적(Expected Stats)**을 계산합니다.

### 4.1. Filtering
전체 데이터셋 $D$에서 조건 $C$ (투수 $P$, 구종 $T$, 타자 유형 $S$)를 만족하는 부분집합 $D'$를 추출합니다.
$$D' = \{d \in D \mid d_{pitcher}=P \land d_{type}=T \land d_{stand}=S\}$$

### 4.2. Metric Calculation
-   **Whiff Rate (헛스윙률):** $\frac{\text{Swinging Strikes}}{\text{Total Swings}} \times 100$
-   **Put Away % (결정구 효율):** 2스트라이크 상황에서 삼진을 잡은 비율.
-   **Hard Hit % (강타 비율):** 타구 속도 95mph 이상인 타구의 비율.