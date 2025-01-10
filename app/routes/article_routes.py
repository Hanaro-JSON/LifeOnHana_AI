from flask import Blueprint, request, jsonify
from .. import vector_service

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
    
    try:
        recommendations = vector_service.get_recommendations(user_id, k)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/record-action', methods=['POST'])
def record_action():
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