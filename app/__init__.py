from flask import Flask
from .services.redis_service import RedisService
from .models.bert_model import BertEmbedding
from .services.vector_service import VectorService
import os
import pymysql  # flask_mysqldb 대신 pymysql 사용
from .api import init_api  # 변경

# 서비스 인스턴스 생성
redis_service = RedisService()
bert_model = BertEmbedding(os.getenv('MODEL_PATH', './Bert'))
vector_service = VectorService(redis_service.client, bert_model)

# MySQL 연결
db = pymysql.connect(
    host='lifeonhana.cxq2u4wk2434.ap-northeast-2.rds.amazonaws.com',
    user='admin',
    password='LifeOnHana1!',
    database='lifeonhana_serverDB',
    cursorclass=pymysql.cursors.DictCursor
)

def create_app():
    app = Flask(__name__)
    
    # API 초기화 (변경)
    init_api(app)
    
    # 블루프린트 등록
    from .routes import article_routes
    app.register_blueprint(article_routes.bp)
    
    # Claude 블루프린트 추가
    from .routes import claude_routes
    app.register_blueprint(claude_routes.bp)
    
    return app 