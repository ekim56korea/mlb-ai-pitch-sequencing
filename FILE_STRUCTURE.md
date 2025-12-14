# 📂 File Structure

프로젝트의 디렉토리 구조와 주요 파일들의 역할입니다. 불필요한 파일은 모두 제거되었습니다.

```text
mlb-ai-pitch-sequencing/
├── README.md               # 프로젝트 메인 설명서
├── ARCHITECTURE.md         # 시스템 아키텍처 문서
├── PHYSICS_LOGIC.md        # 물리 엔진 로직 문서
├── FILE_STRUCTURE.md       # 파일 구조 문서
├── .gitignore              # Git 제외 파일 설정
├── savant_data.csv         # MLB 데이터셋 (Statcast)
├── train_model.py          # AI 모델 학습 스크립트
├── pitch_predictor.pkl     # 학습된 AI 모델 바이너리
│
├── api/                    # [Backend] Python FastAPI Server
│   ├── app.py              # 메인 서버 애플리케이션 (API Endpoints)
│   └── __init__.py
│
└── client/                 # [Frontend] Next.js Application
    ├── package.json        # 프론트엔드 의존성 관리
    ├── next.config.mjs     # Next.js 설정
    ├── tailwind.config.ts  # Tailwind CSS 설정
    ├── public/             # 정적 파일 (이미지 등)
    └── src/
        ├── app/
        │   ├── globals.css # 전역 스타일
        │   └── page.tsx    # 메인 페이지 (Entry Point)
        └── components/     # UI 및 로직 컴포넌트
            ├── SearchModule.tsx    # 메인 컨트롤러 & 상태 관리
            ├── Pitch3D.tsx         # 3D 궤적 및 히트맵 렌더링
            └── AnalyticsCharts.tsx # 2D 분석 차트 (구속/무브먼트)