# 🏗️ System Architecture

## 🔄 Data Flow Pipeline

본 프로젝트는 **단방향 데이터 흐름(Uni-directional Data Flow)** 아키텍처를 채택하여 데이터의 정합성을 보장합니다.

```mermaid
graph LR
    User[User Action] --> Client[Next.js Client]
    Client -->|REST API Request| Server[FastAPI Server]
    
    subgraph Backend System
    Server -->|Query| DataFrame[Pandas In-Memory DB]
    Server -->|Inference| AI[Random Forest Model]
    DataFrame -->|Raw Stats| Server
    AI -->|Probabilities| Server
    end
    
    Server -->|JSON Response| Client
    Client -->|Props Passing| Components[3D & Charts]


🧩 Logical Layers
Presentation Layer (Next.js)

사용자 인터페이스 및 인터랙션 담당.

Three.js를 통한 WebGL 렌더링 수행 (클라이언트 사이드 연산).

Application Layer (FastAPI)

비즈니스 로직 수행 (시뮬레이션 계산, 데이터 필터링).

AI 모델 추론 수행.

데이터 전송 최적화 (필요한 컬럼만 JSON 변환).

Data Layer (Pandas & CSV)

정형 데이터(Structured Data) 저장소.

초기 구동 시 RAM에 로드하여 고속 조회(In-memory caching) 구현.