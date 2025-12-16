# ⚾️ Pitch Commander Pro (v8.0)

> **Deep Learning Based MLB Pitch Sequencing & Analytics Platform**
>
> 10년 치 MLB 빅데이터와 딥러닝(LSTM)을 활용한 투구 예측 및 3D 시뮬레이션 시스템

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20|%20Next.js%20|%20PyTorch%20|%20DuckDB-blueviolet)

## 📖 Project Overview
**Pitch Commander Pro**는 단순한 야구 기록실을 넘어선 **AI 기반 전력 분석 솔루션**입니다.
MLB Statcast 데이터를 **DuckDB**에 적재하여 초고속으로 조회하며, **PyTorch LSTM** 모델을 통해 투수의 투구 패턴을 시계열로 분석하여 다음 공을 예측합니다. 또한, 물리 엔진을 통해 공의 궤적을 3D로 시각화하고 **구종 가치(Run Value)**를 계산하여 투수의 실질적인 위력을 평가합니다.

## ✨ Key Features (핵심 기능)

1.  **🧠 Deep Learning Prediction (LSTM)**
    * 단순 확률 통계가 아닌, 투구의 시퀀스(이전 5개 공의 흐름)를 학습하여 다음 구종을 예측합니다.
    * 상황(볼카운트, 타자 유형)에 따른 투수의 심리적 패턴을 반영합니다.

2.  **📊 Advanced Analytics (Run Value)**
    * MLB 선형 가중치(Linear Weights) 알고리즘을 적용하여 구종별 **Run Value(RV/100)**를 산출합니다.
    * 투수가 해당 구종으로 실점을 얼마나 억제했는지 정량적으로 평가합니다.

3.  **physics-Based 3D Engine**
    * Statcast의 `pfx_x`, `pfx_z` 데이터를 역산하여 마그누스 효과와 중력이 적용된 3D 궤적을 렌더링합니다.
    * Three.js 기반의 실시간 인터랙티브 시뮬레이션을 제공합니다.

4.  **Big Data Infrastructure**
    * **DuckDB**: 300만 건 이상의 대용량 투구 데이터를 로컬 웨어하우스에 구축하여 지연 없는 분석을 제공합니다.
    * **Dockerized**: 백엔드와 프론트엔드가 컨테이너로 완벽하게 격리 및 관리됩니다.

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Three.js (R3F), Recharts |
| **Backend** | Python FastAPI, Pandas, NumPy, PyBaseball |
| **AI & Data** | **PyTorch (LSTM)**, Scikit-learn, **DuckDB**, Joblib |
| **DevOps** | Docker, Docker Compose |

## 🚀 Quick Start (Installation)

이 프로젝트는 Docker Compose를 통해 한 번에 실행됩니다.

```bash
# 1. 저장소 클론
git clone [https://github.com/your-username/pitch-commander-pro.git](https://github.com/your-username/pitch-commander-pro.git)
cd pitch-commander-pro

# 2. 실행 (DB 구축 및 모델 로딩 자동 수행)
docker-compose up --build


Frontend: http://localhost:3000

Backend API: http://localhost:8000/docs