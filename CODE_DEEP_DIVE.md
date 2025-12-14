# 📂 Code Deep Dive & File Manifest

프로젝트의 주요 파일들과 내부 핵심 함수들의 역할을 상세히 기술합니다.

## 1. Backend (`/api`)

### `api/app.py`
백엔드의 심장이자 두뇌입니다. FastAPI 프레임워크를 기반으로 데이터 처리와 AI 추론을 담당합니다.

* **Global Variables**:
    * `global_df`: CSV 데이터를 메모리에 로드한 Pandas DataFrame. 성능을 위해 서버 시작 시 한 번만 로드합니다.
    * `ai_model`: `joblib`으로 로드된 학습된 Random Forest 모델.
* **Key Functions**:
    * `load_matchup(pitcher, batter)`: 투수 이름을 받아 해당 투수의 구종 레퍼토리(`arsenal`)와 모든 투구 위치 데이터(`locations`)를 JSON으로 반환합니다.
    * `predict_pitch(ctx)`: 현재 게임 상황(볼카운트 등)을 받아 AI 모델에 넣고, 다음 구종 확률 상위 3개를 반환합니다.
    * `simulate_outcome(req)`: 특정 구종 선택 시, 과거 데이터를 필터링하여 `whiff_rate`, `hard_hit_rate` 등을 실시간으로 계산합니다.

---

## 2. Frontend (`/client/src/components`)

### `SearchModule.tsx` (Controller)
프론트엔드의 **중앙 관제탑**입니다.
* **Role**: 사용자 입력(검색, 볼카운트 조작)을 처리하고 백엔드 API와 통신합니다.
* **State Management**: `data`(투수 정보), `predictions`(AI 예측값), `simResult`(시뮬레이션 결과) 상태를 관리하고 하위 컴포넌트에 분배합니다.
* **Logic**:
    * `handleSearch()`: 백엔드 `/load/matchup` 호출.
    * `runPrediction()`: 백엔드 `/predict` 호출.
    * `runSimulation()`: AI 예측 결과 클릭 시 `/simulate_outcome` 호출.

### `Pitch3D.tsx` (Visualization Engine)
Three.js(R3F)를 이용한 **3D 렌더링 엔진**입니다.
* **Components**:
    * `Trajectory`: 물리 공식을 적용해 공의 궤적(Curve)을 그립니다. 마우스 오버 시 구속 정보를 표시합니다.
    * `Heatmap3D`: `locations` 데이터를 받아 3D 격자(Grid)를 생성하고 밀도에 따라 색상을 입힙니다. `depthWrite={false}` 설정을 통해 투명도 겹침 문제를 해결했습니다.
    * `StadiumElements`: 마운드, 홈플레이트, 배터 박스 등 경기장 환경을 구축합니다.

### `AnalyticsCharts.tsx` (Data Dashboard)
Recharts 라이브러리를 사용한 **2D 통계 분석 도구**입니다.
* **Role**: 3D로 파악하기 힘든 정량적 데이터를 시각화합니다.
* **Charts**:
    * **Movement Tab**: `ScatterChart`를 사용하여 `pfx_x`(수평)와 `pfx_z`(수직) 변화량을 산점도로 표현합니다.
    * **Velocity Tab**: `BarChart`를 사용하여 구속 구간별 빈도수를 히스토그램으로 표현합니다.

---

## 3. Data Processing

### `train_model.py`
AI 모델을 생성하는 학습 스크립트입니다.
* **Workflow**:
    1.  `savant_data.csv` 로드.
    2.  결측치(NaN) 제거 및 필요한 컬럼(`pitch_type`, `balls`, `strikes` 등) 선택.
    3.  `RandomForestClassifier` 학습 수행.
    4.  `pitch_predictor.pkl` 파일로 직렬화(Serialization)하여 저장.