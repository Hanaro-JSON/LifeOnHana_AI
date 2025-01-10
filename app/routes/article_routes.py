from flask import Blueprint, request, jsonify
from .. import vector_service
import traceback

bp = Blueprint('articles', __name__)

@bp.route('/')
def hello():
    return 'Flask server is running!'

@bp.route('/test_embedding', methods=['POST'])
def test_embedding():
    data = request.json
    text = data.get('text', '')
    try:
        embedding = vector_service.bert.get_embedding(text)
        return jsonify({
            "text": text,
            "embedding_size": len(embedding),
            "embedding_sample": embedding[:5].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/articles', methods=['POST'])
def add_article():
    data = request.json
    try:
        vector_service.store_article(
            article_id=data['id'],
            title=data['title'],
            content=data['content']
        )
        return jsonify({"status": "success"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/articles/similar', methods=['POST'])
def find_similar():
    data = request.json
    try:
        results = vector_service.search_similar(data['query'])
        similar_articles = [{
            "title": doc.title,
            "content": doc.content,
            "similarity": 1 - float(doc.score)
        } for doc in results.docs]
        return jsonify(similar_articles)
    except Exception as e:
        return jsonify({"error": str(e)}), 400 

@bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    k = int(request.args.get('k', 10))
    
    # 컨텍스트 정보 추가
    context = {
        'time_of_day': vector_service._get_time_of_day(),
        'platform': request.args.get('platform', 'web'),
        'location': request.args.get('location', 'unknown')
    }
    
    try:
        recommendations = vector_service.get_recommendations(user_id, context, k)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/articles/<article_id>', methods=['GET'])
def view_article(article_id):
    """기사 조회"""
    try:
        # 기사 조회 로직만 남기고 행동 기록 제거
        article = get_article(article_id)
        return jsonify(article)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/articles/<article_id>/like', methods=['POST'])
def like_article(article_id):
    """기사 좋아요"""
    try:
        # 1. 좋아요 처리
        # 2. 내부적으로 행동 기록
        vector_service.record_user_action(
            user_id=request.headers['user_id'],
            article_id=article_id,
            action_type='like'
        )
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400 

@bp.route('/user/action', methods=['POST'])
def record_action():
    """사용자 행동 기록"""
    data = request.json
    try:
        vector_service.record_user_action(
            user_id=data['user_id'],
            article_id=data['article_id'],
            action_type=data['action_type']
        )
        return jsonify({"status": "success"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400 

@bp.route('/articles/check', methods=['GET'])
def check_articles():
    """저장된 기사 확인"""
    try:
        # Redis에서 모든 기사 키 검색
        articles = []
        for key in vector_service.redis.scan_iter("article:*"):
            if isinstance(key, bytes):
                key = key.decode('utf-8')
                
            article_data = {}
            # 각 필드를 개별적으로 가져와서 처리
            raw_data = vector_service.redis.hgetall(key)
            
            for field, value in raw_data.items():
                field = field.decode('utf-8') if isinstance(field, bytes) else field
                if field == 'embedding':
                    article_data[field] = 'exists' if value else 'missing'
                else:
                    article_data[field] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            articles.append({
                'key': key,
                'data': article_data
            })
        
        return jsonify({
            'article_count': len(articles),
            'articles': articles,
            'vector_dim': vector_service.VECTOR_DIM
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 400 