from flask import Blueprint, jsonify, request, current_app
import os
from app.services.article_service import ArticleService
from ..services.vector_service import VectorService

bp = Blueprint('articles', __name__)
article_service = ArticleService()
vector_service = None

def init_vector_service(app):
    global vector_service
    with app.app_context():
        vector_service = VectorService(app.redis_client, app.bert_model)

def get_vector_service():
    global vector_service
    if vector_service is None:
        vector_service = VectorService(current_app.redis_client, current_app.bert_model)
    return vector_service

ARTICLES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'hanathenext_articles')

@bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """사용자 맞춤 기사 추천 API"""
    try:
        user_id = request.args.get('userId')
        size = request.args.get('size', default=20, type=int)
        seed = request.args.get('seed')

        current_app.logger.info(f"Received userId: {user_id}, size: {size}, seed: {seed}")
        
        
        if not user_id:
            return jsonify({
                "code": 400,
                "status": "BAD_REQUEST",
                "message": "userId가 필요합니다.",
                "data": None
            }), 400
            
        vector_service = get_vector_service()
        recommended_articles = vector_service.get_recommendations(user_id, seed, size)
        
        return jsonify({
            "code": 200,
            "status": "OK",
            "message": "추천 목록 조회 성공",
            "data": {
                "recommendedArticles": recommended_articles
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"추천 API 에러: {str(e)}")
        return jsonify({
            "code": 500,
            "status": "INTERNAL_SERVER_ERROR",
            "message": "서버 에러가 발생했습니다.",
            "data": None
        }), 500

@bp.route('/process-folder', methods=['POST'])
def process_folder():
    try:
        current_app.logger.info(f"작업 폴더 경로: {ARTICLES_FOLDER}")
        service = get_vector_service()
        success, message = article_service.process_folder(ARTICLES_FOLDER)
        
        if success:
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "message": message})
            
    except Exception as e:
        current_app.logger.error(f"Error in process_folder: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}) 