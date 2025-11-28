# Software Requirements Specification (SRS)
**Project:** Pitch Commander Pro (v1.0)
**Date:** 2025-11-28
**Author:** Lead Data Scientist (Your Name)

---

## 1. Introduction (개요)
### 1.1 Purpose (목적)
The purpose of this document is to define the functional and non-functional requirements for the **Pitch Commander Pro** system. This system aims to provide real-time, optimal pitch sequencing recommendations to MLB pitchers and coaching staff.
본 문서는 **Pitch Commander Pro** 시스템의 기능적 및 비기능적 요구사항을 정의합니다. 이 시스템은 MLB 투수와 코칭 스태프에게 실시간으로 최적화된 투구 배합을 추천하는 것을 목적으로 합니다.

### 1.2 Scope (범위)
* **Input:** Live Statcast data (PITCHf/x), Game Context (Inning, Score, Count).
* **Processing:** Physics trajectory simulation, Batter clustering, Game theory optimization.
* **Output:** Next pitch recommendation (Type, Location) via Dashboard UI.

---

## 2. Functional Requirements (기능적 요구사항)

### 2.1 Physics & Mechanics Engine (물리 및 역학 엔진)
* **REQ-PHY-01 (Trajectory Simulation):** The system shall calculate the 3D trajectory of a pitch using 9-parameter fit data (vx0, vy0, vz0, ax, ay, az, etc.).
    * 시스템은 9가지 매개변수 데이터를 사용하여 투구의 3D 궤적을 계산해야 한다.
* **REQ-PHY-02 (Angle Calculation):** The system must compute Vertical Approach Angle (VAA) and Horizontal Approach Angle (HAA) at home plate.
    * 시스템은 홈플레이트 도달 시점의 수직 입사각(VAA)과 수평 입사각(HAA)을 산출해야 한다.

### 2.2 AI Predictive Modeling (AI 예측 모델링)
* **REQ-AI-01 (Outcome Prediction):** The model shall predict the probabilities of [Strike, Ball, In-Play(Out), In-Play(Hit)] for a given pitch.
    * 모델은 주어진 투구에 대해 [스트라이크, 볼, 범타, 안타]의 확률을 예측해야 한다.
* **REQ-AI-02 (Batter Clustering):** The system shall categorize batters into 5 distinct clusters based on their swing/take propensities using K-Means.
    * 시스템은 스윙/테이크 성향에 기반하여 타자를 5개의 클러스터로 분류해야 한다.

### 2.3 Strategy & Game Theory (전략 및 게임 이론)
* **REQ-STR-01 (xRV Calculation):** The system shall calculate the Expected Run Value (xRV) for each potential pitch option.
    * 시스템은 가능한 모든 투구 옵션에 대해 기대 득점 가치(xRV)를 계산해야 한다.
* **REQ-STR-02 (Entropy Mixing):** To prevent predictability, the final recommendation must provide a mixed strategy (probabilities) rather than a deterministic single pitch.
    * 예측 불가능성을 위해, 최종 추천은 단정적인 하나의 구종이 아닌 혼합 전략(확률 분포)을 제공해야 한다.

---

## 3. Non-Functional Requirements (비기능적 요구사항)

### 3.1 Performance (성능)
* **NFR-PER-01 (Latency):** The end-to-end inference time must not exceed **200ms** after receiving data.
    * 데이터 수신 후 최종 추론까지의 시간은 **200ms**를 초과해서는 안 된다.

### 3.2 Explainability (설명 가능성)
* **NFR-XAI-01:** The dashboard must display the 'Risk Score' and 'Reasoning' (e.g., "High Fatigue detected") for the recommended pitch.
    * 대시보드는 추천된 투구에 대한 '리스크 점수'와 '추천 사유'(예: "높은 피로도 감지됨")를 표시해야 한다.

### 3.3 Reliability (신뢰성)
* **NFR-REL-01:** The system must handle missing Statcast data by falling back to the pitcher's historical averages.
    * 시스템은 Statcast 데이터 누락 시 투수의 과거 평균 데이터를 사용하여 예외를 처리해야 한다.

---

## 4. System Constraints (시스템 제약사항)
* **Development Language:** Python 3.9+
* **Interface:** Streamlit (Web Dashboard)
* **Data Source:** PyBaseball (Statcast API)
