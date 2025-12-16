### 3. 📂 FILE_STRUCTURE.md (파일 구조)

```markdown
# 📂 Project File Structure

본 프로젝트는 **Backend (Python/Docker)**와 **Frontend (Next.js)**가 분리된 모노레포 구조입니다.

```text
mlb-ai-pitch-sequencing/
├── docker-compose.yml       # [Core] 컨테이너 오케스트레이션 설정
├── .gitignore
├── README.md
├── TECHNICAL_REPORT.md      # 알고리즘 및 아키텍처 설명서
│
├── backend/                 # [Backend] FastAPI & AI Engine
│   ├── Dockerfile           # 백엔드 이미지 빌드 설정
│   ├── requirements.txt     # Python 의존성 목록
│   ├── app/                 # 애플리케이션 소스 코드
│   │   ├── main.py          # API 엔드포인트 & 비즈니스 로직
│   │   ├── model.py         # PyTorch LSTM 모델 클래스 정의
│   │   └── physics.py       # 3D 궤적 계산 물리 엔진
│   ├── scripts/             # 데이터 및 학습 스크립트
│   │   ├── setup_db.py      # DuckDB 구축 스크립트
│   │   └── train_lstm.py    # AI 모델 학습 스크립트
│   └── data/                # [Data] (Git 제외됨)
│       ├── savant.duckdb    # 대용량 야구 데이터 웨어하우스
│       ├── pitch_lstm.pth   # 학습된 AI 모델 가중치
│       └── encoders.pkl     # 데이터 전처리 객체
│
└── client/                  # [Frontend] Next.js 14 App
    ├── Dockerfile           # 프론트엔드 이미지 빌드 설정
    ├── package.json
    ├── src/
    │   ├── app/             # Page Router
    │   └── components/      # UI Components
    │       ├── SearchModule.tsx    # [Core] 검색, 필터, 상태 관리
    │       ├── Pitch3D.tsx         # [Core] Three.js 3D 렌더링
    │       └── AnalyticsCharts.tsx # [Core] Recharts 데이터 시각화