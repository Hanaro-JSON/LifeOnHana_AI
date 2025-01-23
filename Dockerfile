FROM python:3.8-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    g++ \
    curl \
    automake \
    openjdk-17-jdk \
    python3-dev \
    wget \
    mecab \
    mecab-ipadic-utf8 \
    libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -s

# korpora 직접 설치
RUN pip install git+https://github.com/ko-nlp/Korpora.git

# Korpora 데이터 다운로드 스크립트 생성
COPY download_korpora.py .
RUN python download_korpora.py

# MeCab 설치 확인 스크립트
RUN echo '#!/bin/bash\n\
echo "=== MeCab 설치 확인 ==="\n\
which mecab\n\
echo "=== MeCab 사전 경로 ==="\n\
mecab-config --dicdir\n\
echo "=== MeCab 사전 파일 검색 ==="\n\
find / -name "dicrc" 2>/dev/null\n\
' > /check-mecab.sh && chmod +x /check-mecab.sh

# 스크립트 실행
RUN /check-mecab.sh

# 애플리케이션 코드 복사
COPY . .

# 환경변수는 docker-compose.yml에서 설정됨
EXPOSE 5001

# docker-compose.yml의 command와 일치
CMD ["python", "app.py"]

