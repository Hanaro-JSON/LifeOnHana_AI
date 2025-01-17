from flask import Blueprint, jsonify, current_app
import os
from app.services.article_service import ArticleService

bp = Blueprint('articles', __name__)
article_service = ArticleService()

ARTICLES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'hanathenext_articles')

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