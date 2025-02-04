import redis

class RedisService:
    def __init__(self, host='redis', port=6379):
        self.client = redis.Redis(
            host=host,
            port=port,
            decode_responses=False
        ) 