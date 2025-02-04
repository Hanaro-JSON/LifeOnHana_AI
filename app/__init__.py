from flask import Flask
from .services.redis_service import RedisService
from .models.bert_model import BertEmbedding
from .services.vector_service import VectorService
import os
import pymysql  # flask_mysqldb 대신 pymysql 사용
from .api import init_api  # 변경
from config.settings import MYSQL_CONFIG


# 서비스 인스턴스 생성
redis_service = RedisService()
bert_model = BertEmbedding(os.getenv('MODEL_PATH', './Bert'))
vector_service = VectorService(redis_service.client, bert_model)

# MySQL 연결
db = pymysql.connect(
    host=MYSQL_CONFIG['host'],
    user=MYSQL_CONFIG['user'],
    password=MYSQL_CONFIG['password'],
    database=MYSQL_CONFIG['database'],
    charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
    cursorclass=MYSQL_CONFIG['cursorclass']
)

def create_app():
    app = Flask(__name__)
    
    # 서비스 인스턴스를 앱 객체에 추가
    app.redis_client = redis_service.client
    app.bert_model = bert_model
    
    # API 초기화
    init_api(app)
    
    # 블루프린트 등록
    from .routes import article_routes
    app.register_blueprint(article_routes.bp)
    article_routes.init_vector_service(app)
    
    # Claude 블루프린트 추가
    from .routes import claude_routes
    app.register_blueprint(claude_routes.bp)
    
    # 벡터 서비스 초기화
    vector_service.initialize_vectors()  # 모든 기사 벡터 생성
    app.vector_service = vector_service  # Flask app에 서비스 추가
    
    return app 