import os
import pymysql.cursors

MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'database': os.getenv('MYSQL_DATABASE', 'default_db'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': 0
}

CACHE_CONFIG = {
    'ttl': int(os.getenv('CACHE_TTL', 3600)), 
    'max_size': int(os.getenv('CACHE_MAX_SIZE', 1000))
}

API_KEY = os.getenv("CLAUDE_API_KEY")

VECTOR_DIM = int(os.getenv('VECTOR_DIM', 768))

AB_TEST_GROUPS = {
    'A': {'content': 0.4, 'cf': 0.3, 'time': 0.2, 'diversity': 0.1},
    'B': {'content': 0.3, 'cf': 0.4, 'time': 0.2, 'diversity': 0.1}
}

CONTEXT_WEIGHTS = {
    'morning': {'investment': 1.2, 'market_news': 1.1},
    'afternoon': {'loan': 1.2, 'credit': 1.1},
    'evening': {'savings': 1.2, 'insurance': 1.1}
}