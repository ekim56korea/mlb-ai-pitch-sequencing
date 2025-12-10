# Pitch Commander Pro v4.0: Tactical Edition ⚾️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-red) ![Physics](https://img.shields.io/badge/Physics-Aerodynamics-orange) ![AI](https://img.shields.io/badge/AI-XGBoost-green)

> **"From Physics to Tactics."**
> 
> **MLB 실데이터 연동, 초정밀 공기역학 엔진, 그리고 경기 상황(Context)을 인지하는 전략 AI가 결합된 차세대 투구 의사결정 시스템입니다.**

---

## 🚀 What's New in v4.0 (Tactical Update)

### 1. 🌍 Real-World Data Injection (실데이터 연동)
* **Dynamic Loader:** `PyBaseball`을 통해 투수와 타자의 이름을 입력하면 **최신 Statcast 데이터**를 실시간으로 서버에 로드합니다.
* **Auto-Calibration:** 투수의 실제 구종, 평균 구속, 무브먼트 데이터를 분석하여 시뮬레이션 초기값을 자동으로 보정합니다.

### 2. 🏟️ Hyper-Physics Engine V3 (환경 변수 적용)
* **Environmental Normalization:** 경기장의 **기온(Temp), 해발고도(Elevation), 습도(Humidity)**를 입력받아 공기 밀도($\rho$)를 동적으로 계산합니다.
* **Aerodynamics:** 쿠어스 필드(Coors Field)와 펫코 파크(Petco Park)에서의 변화구 궤적 차이를 1:1로 시뮬레이션합니다.

### 3. 📊 Pitching+ Metrics (구위 평가 AI)
* **Stuff+ Model:** `XGBoost` 기반 머신러닝 모델이 투구의 물리적 제원(속도, 회전, 무브먼트, 익스텐션)을 분석하여 **객관적인 구위 점수(Stuff+)**를 산출합니다.
* **Actionable Insight:** "이 공은 리그 평균 대비 상위 10%의 위력을 가집니다"와 같은 직관적인 지표를 제공합니다.

### 4. 🧠 Context-Aware Strategy (상황별 전략)
* **Tactical AI:** 볼카운트, 아웃 카운트, 주자 상황, 점수 차 등 **경기 맥락(Context)**을 입력받아 최적의 전략을 수립합니다.
* **Example:** "2사 만루 위기 상황"에서는 땅볼 유도(Sinkers down)를, "2스트라이크"에서는 헛스윙 유도(High Fastball)를 추천합니다.

---

## 🛠️ System Architecture

```mermaid
graph TD
    User[Coach/User] -->|Input: Name & Context| UI[Streamlit Dashboard]
    UI -->|API Request| API[FastAPI Server]
    
    subgraph Core Engine
        API -->|Fetch Data| DL[Data Loader]
        DL -->|Statcast| WEB[(MLB Server)]
        API -->|Calc Physics| PHY[Physics V3]
        API -->|Eval Quality| MET[Metrics Engine]
        API -->|Decision| STR[Strategy Engine]
    end
    
    PHY -->|Trajectory & VAA/HAA| UI
    MET -->|Stuff+ Score| UI
    STR -->|Best Pitch & Target| UI


    💻 Quick Start
1. 환경 설정

Bash
# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 필수 라이브러리 설치 (XGBoost 포함)
pip install -r requirements.txt
2. AI 모델 학습 (최초 1회 필수)

구위 평가 모델(Stuff+)을 생성합니다.

Bash
python scripts/train_stuff_plus.py
3. 시스템 가동

두 개의 터미널을 열어 각각 실행합니다.

Terminal 1: 백엔드 서버 (Brain)

Bash
uvicorn api.app:app --reload
Terminal 2: 대시보드 (Cockpit)

Bash
streamlit run ui/dashboard.py
📈 Tech Stack
Language: Python 3.9+

Core: Pandas, NumPy, SciPy (ODE Solver)

AI/ML: XGBoost, Scikit-learn

Web/API: FastAPI, Uvicorn

Frontend: Streamlit, Plotly (3D Visualization)

Data Source: PyBaseball (Statcast)