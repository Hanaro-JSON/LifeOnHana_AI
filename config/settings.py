import os

# 설정 파일 생성
VECTOR_DIM = 768

AB_TEST_GROUPS = {
    'A': {'content': 0.4, 'cf': 0.3, 'time': 0.2, 'diversity': 0.1},
    'B': {'content': 0.3, 'cf': 0.4, 'time': 0.2, 'diversity': 0.1}
}

CONTEXT_WEIGHTS = {
    'morning': {'investment': 1.2, 'market_news': 1.1},
    'afternoon': {'loan': 1.2, 'credit': 1.1},
    'evening': {'savings': 1.2, 'insurance': 1.1}
}

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

CACHE_CONFIG = {
    'ttl': 3600,  # 1시간
    'max_size': 1000
}

# MySQL(RDS) 설정
MYSQL_CONFIG = {
    'host': 'lifeonhana.cxq2u4wk2434.ap-northeast-2.rds.amazonaws.com',
    'user': 'admin',
    'password': 'LifeOnHana1!',
    'database': 'lifeonhana_serverDB',
    'charset': 'utf8mb4',
    'cursorclass': 'DictCursor'
} 