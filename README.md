# ⚾️ MLB AI Pitch Sequencing & Visualization Platform

> **Advanced Baseball Analytics System powered by 3D Physics & Machine Learning**

이 프로젝트는 MLB Statcast 데이터를 활용하여 투수의 투구 궤적을 물리적으로 시뮬레이션하고, 머신러닝(Random Forest)을 통해 최적의 볼 배합을 예측하며, 타격 결과를 시뮬레이션하는 **데이터 기반 야구 전력 분석 플랫폼**입니다.

![Project Status](https://img.shields.io/badge/Project-Complete-brightgreen)
![Tech Stack](https://img.shields.io/badge/Next.js-FastAPI-blueviolet)

## ✨ Core Features (핵심 기능)

1.  **Physics-Based 3D Visualization**: 마그누스 효과(Magnus Effect)와 중력을 반영한 리얼타임 투구 궤적 렌더링.
2.  **Volumetric Heatmap**: 투구 위치 데이터(Point Cloud)를 복셀(Voxel) 격자로 변환하여 시각화한 3D 히트맵.
3.  **AI Pitch Prediction**: 상황(볼카운트, 타자 유형 등)에 따른 투수의 다음 구종 예측 (Accuracy > 70%).
4.  **Deep Analytics Dashboard**: 구속(Velocity) 및 무브먼트(Movement) 정밀 분석 차트.
5.  **Outcome Simulator**: 특정 구종 선택 시 예상되는 헛스윙률(Whiff%), 강타 비율(Hard Hit%) 시뮬레이션.

## 🛠️ Tech Stack

-   **Frontend**: Next.js 14 (App Router), React, Three.js (R3F), Recharts, Tailwind CSS
-   **Backend**: Python FastAPI, Pandas, NumPy
-   **AI/ML**: Scikit-learn (Random Forest), Joblib
-   **Data**: MLB Statcast (Baseball Savant)

## 🚀 Quick Start

### Backend (Python)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install fastapi uvicorn pandas scikit-learn joblib

# AI 모델 학습 (최초 1회)
python train_model.py

# 서버 실행
uvicorn api.app:app --reload