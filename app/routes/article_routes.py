from flask import Blueprint, jsonify, current_app, request
import os
from app.services.article_service import ArticleService
from app.services.vector_service import VectorService

bp = Blueprint('articles', __name__)
article_service = ArticleService()
vector_service = VectorService(current_app.redis_client, current_app.bert_model)

ARTICLES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'hanathenext_articles')

@bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        user_id = request.args.get('userId')
        seed = request.args.get('seed')
        k = int(request.args.get('size', 50))
        
        if not user_id:
            return jsonify({
                "status": "error",
                "message": "userId is required"
            }), 400
            
        recommended_articles = vector_service.get_recommendations(user_id, seed, k)
        
        return jsonify({
            "status": "success",
            "data": recommended_articles
        })
        
    except Exception as e:
        current_app.logger.error(f"추천 API 에러: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/process-folder', methods=['POST'])
def process_folder():
    try:
        current_app.logger.info(f"작업 폴더 경로: {ARTICLES_FOLDER}")
        success, message = article_service.process_folder(ARTICLES_FOLDER)
        
        if success:
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "message": message})
            
    except Exception as e:
        current_app.logger.error(f"Error in process_folder: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}) 