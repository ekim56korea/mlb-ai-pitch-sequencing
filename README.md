# mlb-ai-pitch-sequencing
"Enterprise-grade real-time tactical decision support system for MLB pitchers based on Statcast physics and Game Theory.

# Pitch Commander Pro: MLB Real-time Tactical Decision System ⚾️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Framework](https://img.shields.io/badge/Framework-Streamlit-red) ![License](https://img.shields.io/badge/License-MIT-green)

> **Enterprise-grade AI solution for optimizing pitch sequencing using Statcast physics and Game Theory.**
>
> **Statcast 물리학과 게임 이론(Game Theory)을 결합하여 투구 배합을 최적화하는 엔터프라이즈급 실시간 AI 솔루션입니다.**

---

## 1. Project Charter (프로젝트 정의)

### 🧐 Business Context & Problem (비즈니스 배경 및 문제)
In modern baseball, batters enter the box with pitcher patterns already memorized through static data analysis. Simply throwing the "best stuff" leads to a drastic increase in OPS allowed during the **Times Through Order (TTO)** penalty.
현대 야구에서 타자들은 정적 데이터 분석을 통해 투수의 패턴을 이미 학습하고 타석에 들어섭니다. 단순히 '구위가 좋은 공'을 던지는 것만으로는 타순이 한 바퀴 돌았을 때(Times Through Order) 피OPS가 급증하는 문제를 해결할 수 없습니다.

### 💡 Solution (해결책)
**Pitch Commander Pro** integrates **Statcast 9-Parameter Physics** with **Nash Equilibrium Game Theory**. It recommends pitch sequences that maximize entropy to break batter predictions while optimizing **xRV (Expected Run Value)** based on physical constraints.
**Pitch Commander Pro**는 **Statcast 9-Parameter 물리 엔진**과 **내쉬 균형 게임 이론**을 통합했습니다. 이는 타자의 예측을 깨기 위해 엔트로피를 극대화하는 동시에, 물리적 제약 조건을 고려하여 **기대 득점 가치(xRV)**를 최소화하는 투구 시퀀스를 실시간으로 추천합니다.

### 🏆 Key Success Metrics (핵심 성과 지표)
* **xRV Reduction:** Achieve -0.5 runs per 9 innings compared to league average. (9이닝당 기대 득점 -0.5점 감소 달성)
* **TTO Defense:** Suppress the OPS increase during the 3rd time through the order by 30%. (3번째 타석 상대 시 피OPS 상승폭 30% 억제)

---

## 2. System Architecture (시스템 아키텍처)

The system operates on a modular **4-Layer Architecture** to ensure scalability and maintainability.
이 시스템은 확장성과 유지보수성을 보장하기 위해 모듈화된 **4계층 아키텍처**로 작동합니다.

```mermaid
graph TD
    A[Data Ingestion] -->|Cleaned Data| B(Physics Engine)
    B -->|Trajectory & VAA/HAA| C(Predictive Models)
    C -->|Probabilities| D(Strategy Engine)
    D -->|Optimal Decision| E[Client Dashboard]
