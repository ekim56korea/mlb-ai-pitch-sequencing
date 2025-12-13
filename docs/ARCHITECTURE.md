```markdown
# 🏗️ System Architecture

Pitch Commander Pro v7.0은 **Micro-Modular Architecture**를 채택하여 각 엔진이 독립적이면서도 유기적으로 연결되도록 설계되었습니다. 특히 비용 절감을 위해 **로컬 자원 최적화(On-Premise Optimization)**에 집중했습니다.

## 🔄 Data Flow Pipeline

```mermaid
graph TD
    A[MLB Statcast (PyBaseball)] -->|Raw Data| B(Data Preprocessor);
    B -->|Cleaned Data| C{Hybrid Storage};
    C -->|Metadata & SQL| D[(SQLite DB - WAL Mode)];
    C -->|Cold Data (Fast Read)| E[Parquet Cache];
    
    subgraph "Core Engines (API Server)"
    F[DataLoader] -->|Async IO| C;
    G[Physics Engine] -->|ODEs| H[Trajectory Sim];
    I[AI Models] -->|Bayesian/RF| J[Strategy Engine];
    K[Metrics Engine] -->|Stuff+/Location+| L[Scoring];
    end
    
    subgraph "Presentation (Client)"
    M[Streamlit Dashboard] -->|HTTP/JSON| N[FastAPI Gateway];
    N --> F & G & J & K;
    end
🚀 Optimization Strategies (Zero-Cost Engineering)
1. Hybrid Data Storage

SQLite (WAL Mode):

메타데이터 관리 및 복잡한 관계형 쿼리에 사용합니다.

PRAGMA journal_mode=WAL을 적용하여 읽기/쓰기 동시성을 확보했습니다.

자주 조회되는 pitcher, batter 컬럼에 B-Tree 인덱스를 적용했습니다.

Parquet Caching:

수천 건의 과거 투구 데이터(Cold Data)는 컬럼 기반 압축 포맷인 Parquet로 저장합니다.

pandas.read_parquet을 사용하여 DB 조회 대비 10배 빠른 로딩 속도를 구현했습니다.

2. Asynchronous Processing

FastAPI Async/Await:

데이터 다운로드 및 로딩과 같은 I/O Bound 작업은 asyncio.to_thread를 통해 메인 루프를 차단하지 않고 별도 스레드에서 처리합니다.

Streamlit Caching:

@st.cache_data를 활용하여 동일한 매치업 분석 요청에 대해 API 호출을 생략하고 메모리에서 결과를 즉시 반환합니다.

3. Lightweight AI

무거운 딥러닝 프레임워크(TensorFlow/PyTorch) 대신, Scikit-learn과 Numpy 기반의 수치 연산으로 AI 모델을 경량화하여 CPU 환경에서도 실시간 추론이 가능합니다.



---