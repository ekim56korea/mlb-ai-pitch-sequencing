# 1. 베이스 이미지 설정 (Python 3.10 슬림 버전 권장 - 가볍고 호환성 좋음)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /code

# 3. 시스템 의존성 설치 (필요한 경우)
# (pybaseball이나 일부 ML 라이브러리를 위해 필요한 기본 도구들)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
# 캐시 효율성을 위해 requirements.txt를 먼저 복사합니다.
COPY requirements.txt .

# --no-cache-dir 옵션으로 이미지 크기를 줄입니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
# app 폴더와 scripts 폴더를 컨테이너 내부로 복사합니다.
COPY ./app ./app
COPY ./scripts ./scripts

# (선택) 데이터 폴더가 있다면 복사 (보통은 볼륨으로 연결하지만 초기 파일이 필요하다면)
# COPY ./data ./data 

# 6. 포트 노출 (FastAPI 기본 포트)
EXPOSE 8000

# 7. 실행 명령어
# app 폴더 안의 main.py 파일 내의 app 객체를 실행
# --reload: 코드 수정 시 자동 재시작 (개발용)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]