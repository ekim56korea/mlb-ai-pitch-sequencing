# ⚾️ Pitch Commander Pro v7.0 (Zero-Cost Edition)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red) ![License](https://img.shields.io/badge/License-MIT-grey)

**Pitch Commander Pro**는 고가의 트래킹 장비(Trackman)나 유료 데이터 피드 없이, **순수 수학(Math)과 효율적인 엔지니어링**만으로 엔터프라이즈급 야구 분석을 구현한 AI 시스템입니다.

MLB Statcast 데이터를 기반으로 물리적 궤적을 시뮬레이션하고, 베이지안 추론과 강화학습(RL) 로직을 결합하여 최적의 투구 시퀀스를 제안합니다.

## 🌟 Key Features (The "Zero-Cost" Innovation)

### 1. 🌪️ Hyper-Physics Engine (물리 엔진)
* **Reverse Engineering:** `PFX`와 `Spin Rate`만으로 **회전 효율(Efficiency)**과 **자이로 각도(Gyro Angle)**를 수학적으로 역산합니다.
* **Collision Physics:** 앨런 네이선(Alan Nathan) 교수의 충돌 모델을 구현하여 타격 시 **예상 비거리와 타구 속도**를 시뮬레이션합니다.
* **Environmental Factors:** 기온, 습도, 고도 및 **3D 바람장(Wind Field)**이 궤적에 미치는 영향을 계산합니다.

### 2. 🧠 Deep Intelligence Strategy (AI 전략)
* **Mind Reading:** 베이지안 추론(Bayesian Inference)을 통해 타자의 **카운트별 노림수(Guess Hitting)**를 예측합니다.
* **Context Awareness:** **RE24 Matrix**를 내장하여 경기 상황의 중요도(**Leverage Index**)를 계산하고, 위기 상황에 맞는 전략을 수립합니다.
* **Swing Probability:** 타자가 배트를 낼 확률을 경량 ML 모델로 예측합니다.

### 3. 🏟️ Volumetric Analytics (입체 시각화)
* **Ghost Trails:** 직전 투구와 현재 투구의 궤적을 겹쳐 보여주어 **터널링(Tunneling)** 효과를 분석합니다.
* **3D Hot Zones:** 타자의 약점을 단순한 2D 존이 아닌, 구속-무브먼트 공간(3D)에서 시각화합니다.

### 4. ⚡ Extreme Performance (성능 최적화)
* **Hybrid Storage:** SQLite(WAL Mode)와 Parquet 캐싱을 결합하여 데이터 로딩 속도를 **0.1초** 대로 단축했습니다.
* **Async IO:** FastAPI의 비동기 처리를 통해 시뮬레이션 중에도 서버가 멈추지 않습니다.

## 🛠️ Installation & Setup

1. **저장소 클론**
   ```bash
   git clone [https://github.com/ekim56korea/mlb-ai-pitch-sequencing.git](https://github.com/ekim56korea/mlb-ai-pitch-sequencing.git)
   cd mlb-ai-pitch-sequencing
필수 라이브러리 설치

Bash
pip install -r requirements.txt
▶️ How to Run
이 시스템은 **Backend(API)**와 **Frontend(Dashboard)**가 분리되어 있습니다. 두 개의 터미널에서 각각 실행해주세요.

Terminal 1: API Server

Bash
# 비동기 처리를 위해 4개의 워커 프로세스 가동
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
Terminal 2: Dashboard

Bash
# Streamlit 대시보드 실행
streamlit run ui/dashboard.py
📂 Documentation
Architecture Overview

Physics Logic Details

AI Strategy Algorithm

⚠️ Note
이 프로젝트는 학습 및 연구 목적으로 개발되었습니다. 사용된 데이터는 pybaseball 라이브러리를 통해 수집되며, 상업적 이용 시 MLB 데이터 라이선스 정책을 확인해야 합니다.


---