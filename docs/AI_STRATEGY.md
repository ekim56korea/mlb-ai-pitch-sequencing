# 🧠 AI Strategy & Decision Making

Pitch Commander Pro는 단순한 통계 확률이 아닌, **맥락(Context)**과 **심리(Psychology)**를 결합한 하이브리드 의사결정 트리를 사용합니다.

## 1. Deep Intelligence Pipeline

```mermaid
graph LR
    A[Game Context] --> B(Leverage Index Calc);
    C[Batter History] --> D(Guess Hitting Model);
    C --> E(Swing/Take Model);
    
    B & D & E --> F{Strategy Engine};
    F --> G[Recommended Sequence];
    
2. Core Algorithms
A. Guess Hitting Model (타자 노림수 예측)

알고리즘: 베이지안 추론 (Bayesian Inference)

원리: P(Guess∣Count)∝P(Count∣Guess)×P(Guess)

설명: MLB 평균 타자들의 카운트별 노림수(사전 확률)를 타자의 과거 이력(우도)으로 보정하여, 현재 타자가 특정 구종을 노리고 있을 확률을 계산합니다.

B. Swing Probability (반응성 예측)

알고리즘: Random Forest (경량 ML) & Heuristics

Input: 투구 위치(Plate X/Z), 구종, 볼카운트

Output: 타자가 배트를 낼 확률 (0~100%)

전략적 활용:

Whiff 유도: 스윙 확률 > 70% 이면서 존 바깥인 공 추천.

Freeze (루킹) 유도: 스윙 확률 < 30% 이면서 존 안쪽인 공 추천.

C. Sabermetrics Context (맥락 인식)

Leverage Index (LI):

RE24 Matrix를 기반으로 현재 상황의 승부처 지수를 계산합니다.

Critical (LI > 3.0): 실험적 투구를 배제하고, 기대 득점(Run Value) 억제력이 가장 높은 구종/코스를 강제합니다.

Garbage (LI < 0.7): 타자의 약점 데이터를 수집하기 위한 과감한 투구를 제안합니다.

3. XAI (Explainable AI)
AI는 추천 결과만 내놓지 않고, **"Reasoning(근거)"**를 자연어로 생성합니다.

예: "타자가 직구(Guess: 85%)를 노리고 있으나, 2스트라이크 상황이므로 스윙 확률이 높은 바깥쪽 슬라이더로 헛스윙을 유도합니다."

**[복사 종료]**