FROM python:3.8-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# korpora 직접 설치
RUN pip install git+https://github.com/ko-nlp/Korpora.git

# Korpora 데이터 다운로드 스크립트 생성
COPY download_korpora.py .
RUN python download_korpora.py

# 애플리케이션 코드 복사
COPY . .

# 환경변수는 docker-compose.yml에서 설정됨
EXPOSE 5001

# docker-compose.yml의 command와 일치
CMD ["python", "app.py"]