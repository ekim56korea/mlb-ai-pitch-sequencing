


# 🏛️ Technical Deep Dive & Algorithms

**Project:** Pitch Commander Pro (v8.0)  
**Author:** (Your Name)  
**Date:** 2025-12-16

이 문서는 **Pitch Commander Pro**에 적용된 물리 엔진, 인공지능 모델, 그리고 데이터 분석 알고리즘의 상세 로직을 기술합니다. 이 프로젝트는 단순한 통계 조회를 넘어, **Deep Learning (LSTM)**과 **Physics Simulation**을 결합하여 실제 야구 경기의 불확실성을 모델링했습니다.

---

## 1. System Architecture (시스템 아키텍처)

본 시스템은 **Docker** 컨테이너 환경 위에서 **MSA(Microservices Architecture)**에 가까운 구조로 설계되었습니다. 대용량 데이터 처리와 실시간 추론을 분리하여 성능을 최적화했습니다.

```mermaid
graph TD
    User[Web Client] <-->|HTTP/JSON| Client[Next.js Container]
    Client <-->|API Request| Server[FastAPI Container]
    
    subgraph Backend System
        Server -->|SQL Query| DB[(DuckDB Warehouse)]
        Server -->|Inference| AI[PyTorch LSTM Model]
        Server -->|Analysis| Logic[Run Value & Physics Engine]
    end
````

### 1.1 Key Components

  * **Data Layer (DuckDB):** 10년 치(약 300만 개)의 MLB Statcast 데이터를 **DuckDB** (`savant.duckdb`)에 컬럼 기반(Columnar)으로 저장하여, 기존 Pandas CSV 로딩 방식 대비 조회 속도를 획기적으로 단축했습니다.
  * **AI Layer (PyTorch):** 학습된 모델(`pitch_lstm.pth`)과 전처리 객체(`encoders.pkl`)를 서버 시작 시 메모리에 상주(Warm-up)시켜, API 요청 시 0.01초 이내에 추론(Inference) 결과를 반환합니다.
  * **Containerization:** Frontend(Node.js)와 Backend(Python)가 독립된 Docker 컨테이너에서 실행되며, `docker-compose`를 통해 네트워크로 연결됩니다.

-----

## 2\. Deep Learning Algorithm: Pitch LSTM

기존의 Random Forest(단일 시점 분석)는 투구의 "맥락"을 읽지 못한다는 한계가 있었습니다. 이를 극복하기 위해 **시계열(Time-Series) 데이터** 처리에 특화된 **LSTM(Long Short-Term Memory)** 네트워크를 도입했습니다.

### 2.1 Model Architecture

  * **Input Layer:** 최근 5개의 투구 시퀀스 (Vector Size: 6)
      * Features: `[Release Speed, Plate X, Plate Z, Balls, Strikes, Batter Stand]`
  * **Hidden Layers:**
      * LSTM Layer (128 units, 2 stacked layers) - 시퀀스 패턴 학습
      * Dropout (0.2) - 과적합(Overfitting) 방지
  * **Output Layer:** Fully Connected Layer -\> Softmax -\> 각 구종별 확률(Probability)

### 2.2 Sequence Processing Logic

투수는 직전 공이 무엇이었느냐에 따라 다음 공을 결정합니다.

$$ h_t = f(h_{t-1}, x_t) $$

이 수식에 따라, 현재의 투구 예측($h_t$)은 과거의 투구 흐름($h_{t-1}$)을 기억(Memory)하고 있는 상태에서 이루어집니다. 이를 통해 "직구-직구 다음엔 변화구"와 같은 볼 배합 심리를 모델링합니다.

-----

## 3\. Advanced Metrics: Run Value (구종 가치)

단순히 구속이나 무브먼트만으로는 투수의 위력을 알 수 없습니다. 이를 해결하기 위해 MLB의 **Run Value (RV)** 시스템을 자체 알고리즘으로 구현했습니다.

### 3.1 Linear Weights (선형 가중치) 알고리즘

MLB의 기대 득점(Run Expectancy) 변화량을 기반으로 각 투구 이벤트에 점수를 부여합니다. (2023 MLB 기준 근사치 적용)

| Event | Run Value | 의미 |
| :--- | :--- | :--- |
| **Ball** | +0.06 | 타자에게 약간 유리 |
| **Strike** | -0.06 | 투수에게 약간 유리 |
| **Strikeout** | -0.27 | 투수에게 매우 유리 (아웃) |
| **Single** | +0.48 | 1루타 허용 |
| **Home Run** | +1.40 | 투수에게 치명적 (대량 실점) |

### 3.2 Calculation Logic

백엔드 연산 과정에서 모든 투구 데이터에 대해 아래 수식을 적용합니다.

1.  **개별 투구 가치 산출:**
    $$ RV_{pitch} = \sum (Weights \times Events) $$
2.  **100구당 가치 정규화 (Normalization):**
    $$ RV/100 = \frac{\sum RV_{pitch}}{Count} \times 100 $$

<!-- end list -->

  * **해석:** 결과가 \*\*음수(-)\*\*일수록 투수가 실점을 억제했다는 의미로, 위력적인 구종임을 뜻합니다. (예: RV/100이 -1.5라면, 100구를 던졌을 때 1.5점의 실점을 막아낸 것과 같습니다.)

-----

## 4\. Physics Engine: 3D Trajectory

야구공의 궤적을 그리기 위해 Statcast의 `pfx` 데이터를 역산하여 물리 엔진을 구현했습니다. 단순한 베지에 곡선이 아닌, **실제 물리학 공식**을 사용합니다.

### 4.1 Kinematics (운동학) 역산

Statcast는 공의 최종 무브먼트(`pfx_x`, `pfx_z`)를 제공합니다. 우리는 이를 통해 가속도($a$)를 역산출합니다.

  * **수평 가속도 ($a_x$):**
    $$ a_x = \frac{2 \times (-pfx\_x)}{t^2} $$
    *(좌우 반전 및 단위 변환 적용)*
  * **수직 가속도 ($a_z$):** 중력($g$)과 마그누스 효과에 의한 양력(Lift)의 합력
    $$ a_z = -g + \frac{2 \times pfx\_z}{t^2} $$

### 4.2 Simulation Loop

구해진 가속도를 등가속도 운동 방정식에 대입하여 0.01초 단위의 위치 좌표 $P(t)$를 생성하고, 이를 Three.js의 `TubeGeometry`로 렌더링합니다.

$$ P(t) = P_0 + v_0 t + \frac{1}{2} a t^2 $$

이 과정을 통해 투수가 던진 공이 홈플레이트까지 날아가는 0.4초의 순간을 3D 공간에 정확하게 재현합니다.

```
```
