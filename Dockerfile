# 1. 베이스 이미지 (가볍고 빠른 Python 3.9 사용)
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (C 컴파일러 등)
# pandas나 numpy 설치 시 필요한 경우가 있음
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . .

# 6. 포트 개방 (FastAPI: 8000, Streamlit: 8501)
EXPOSE 8000
EXPOSE 8501

# 7. 실행 명령 (스크립트로 제어 예정이므로 여기선 대기)
CMD ["bash"]