# Test Plan & Traceability Matrix (테스트 계획 및 추적 매트릭스)

**Project:** Pitch Commander Pro
**Version:** 1.0
**Document Status:** Draft

---

## 1. Test Strategy (테스트 전략)

### 1.1 Unit Testing (단위 테스트)
* **Scope:** 개별 함수 및 클래스 단위의 로직 검증.
* **Tools:** `pytest` framework.
* **Key Targets:**
    * 물리 엔진의 궤적 계산 수식 정확성 (Physics Accuracy).
    * 데이터 로더의 예외 처리 (Null 값, 파일 없음 등).
    * 전략 엔진의 확률 계산 합계 (Sum of probabilities = 1.0).

### 1.2 Integration Testing (통합 테스트)
* **Scope:** 모듈 간 데이터 흐름 검증.
* **Scenario:** Data Loader $\rightarrow$ Physics Engine $\rightarrow$ AI Model $\rightarrow$ Dashboard.
* **Key Targets:**
    * Statcast 데이터가 물리 엔진에 입력되었을 때 올바른 9-Param 값이 추출되는가?
    * AI 모델의 추론 결과가 대시보드 UI에 지연 없이(Latency check) 표시되는가?

---

## 2. Requirement Traceability Matrix (RTM, 요구사항 추적 매트릭스)

SRS에 정의된 요구사항(ID)과 이를 검증할 테스트 케이스(TC)를 매핑합니다.

| Req ID | Feature | Test Case ID | Test Scenario | Expected Result |
| :--- | :--- | :--- | :--- | :--- |
| **REQ-PHY-01** | Trajectory Sim | **TC-PHY-001** | **Standard Input:** 릴리스 포인트(2.0, 6.0), 구속 95mph, 회전수 2200rpm 입력. | 홈플레이트(y=1.417) 도달 시 z 좌표가 중력(-32.174) 반영 범위 내 존재해야 함. |
| **REQ-PHY-01** | Trajectory Sim | **TC-PHY-002** | **Zero/Negative Velo:** 구속 0 또는 음수 입력 시. | `ValueError` 발생 및 시스템 종료 방지 (Graceful Failure). |
| **REQ-AI-01** | Outcome Pred | **TC-AI-001** | **Shape Check:** 모델 입력 벡터(Features)의 차원 확인. | 모델이 요구하는 Feature 개수(15개)와 일치해야 함. |
| **REQ-AI-02** | Batter Cluster | **TC-AI-002** | **Clustering:** 100명의 타자 데이터 입력. | 각 타자가 0~4번 클러스터 중 하나로 반드시 할당되어야 함. |
| **REQ-STR-02** | Entropy Mixing | **TC-STR-001** | **Diversity:** 동일 상황(2스트라이크) 100회 시뮬레이션. | 특정 구종 추천 비율이 90%를 초과하지 않아야 함(혼합 전략 작동). |
| **NFR-PER-01** | Latency | **TC-PER-001** | **Speed Test:** `time.time()`으로 전체 파이프라인 실행 시간 측정. | 데이터 입력부터 결과 리턴까지 **0.2초(200ms)** 이내 완료. |
| **NFR-REL-01** | Reliability | **TC-REL-001** | **Missing Data:** 입력 데이터 중 `release_spin_rate`가 NaN인 경우. | 해당 투수의 시즌 평균 회전수로 대체하여 추론 수행. |

---

## 3. Test Environment (테스트 환경)
* **OS:** Windows 10/11 or macOS (Local), Ubuntu 20.04 (CI/CD Server)
* **Python Version:** 3.9.13
* **Required Libraries:** `pytest`, `pytest-cov` (Coverage 측정용)
