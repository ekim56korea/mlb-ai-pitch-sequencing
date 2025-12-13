```markdown
# 🌪️ Physics Engine Logic

**Hyper-Physics Engine v4.0**은 트래킹 레이더(Trackman/Hawk-Eye) 없이도 투구의 3차원 궤적과 회전 특성을 시뮬레이션하기 위해 고안되었습니다.

## 1. 3-DOF Trajectory Simulation
공기역학적 힘을 고려하여 상미분방정식(ODEs)을 풉니다.

$$
\vec{F}_{total} = \vec{F}_{gravity} + \vec{F}_{drag} + \vec{F}_{magnus} + \vec{F}_{ssw}
$$

* **항력 (Drag Force):**
  * 공의 속도와 회전수에 따라 변하는 가변 항력 계수($C_d$)를 적용합니다.
  * Drag Crisis 모델을 단순화하여 고속 투구 시 항력 감소를 반영했습니다.
* **마그누스 힘 (Magnus Force):**
  * 스핀 벡터($\vec{\omega}$)와 속도 벡터($\vec{v}$)의 외적($\vec{\omega} \times \vec{v}$)으로 양력을 계산합니다.

## 2. Reverse Engineering (수학적 역산)
실제 측정된 무브먼트(PFX) 데이터를 역산하여 숨겨진 물리 파라미터를 추정합니다.

### A. 자이로 각도 (Gyro Degree) 추정
트래킹 데이터 없이 자이로 스핀을 추정하는 로직입니다.

1. **이론상 최대 양력($C_{L,max}$)** 계산: 순수 백스핀일 때 발생 가능한 최대 무브먼트 산출.
2. **실제 양력($C_{L,actual}$)** 계산: `pfx_x`, `pfx_z` 데이터를 통해 실제 작용한 가속도 산출.
3. **회전 효율($\eta$)** 도출: $\eta = C_{L,actual} / C_{L,max}$
4. **자이로 각도($\theta_{gyro}$)**: $\theta_{gyro} = \arccos(\eta)$

### B. SSW (Seam-Shifted Wake)
회전 효율이 100%가 아닐 때(자이로 성분 존재 시), 손실된 에너지가 측면 항력(Side Force)으로 전환되어 스위퍼나 싱커의 궤적을 만드는 현상을 모델링했습니다.

## 3. Collision Physics (충돌 모델)
Alan Nathan 교수의 충돌 방정식을 적용하여 타격 결과를 예측합니다.

* **타구 속도 (Exit Velocity):**
  $$V_{exit} = q \cdot V_{pitch} + (1+q) \cdot V_{bat}$$
  * $q$: 반발 계수 (나무 배트 기준 약 0.2)
  * $V_{bat}$: 타격 위치에 따른 유효 배트 스피드 보정

* **비거리 (Distance):**
  발사각(LA)과 타구 속도, 그리고 공기 저항을 고려한 포물선 운동 방정식을 사용합니다.