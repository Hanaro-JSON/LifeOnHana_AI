FROM python:3.8-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install protobuf

# 애플리케이션 코드 복사
COPY . .

# 환경변수는 docker-compose.yml에서 설정됨
EXPOSE 5000

# docker-compose.yml의 command와 일치
CMD ["python", "app.py"]