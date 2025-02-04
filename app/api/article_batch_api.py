from flask import Blueprint, jsonify, current_app, g
import pymysql
from datetime import datetime
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict

# API 네임스페이스 생성
article_batch_bp = Blueprint('article_batch', __name__)

def get_db():
    if 'db' not in g:
        g.db = pymysql.connect(
            host='lifeonhana.cxq2u4wk2434.ap-northeast-2.rds.amazonaws.com',
            user='admin',
            password='LifeOnHana1!',
            database='lifeonhana_serverDB',
            cursorclass=pymysql.cursors.DictCursor
        )
    return g.db

def generate_s3_key(title: str, suffix: str) -> str:
    """제목을 기반으로 S3 키 생성"""
    hash_object = hashlib.md5(title.encode())
    hash_value = hash_object.hexdigest()[:8]
    safe_title = re.sub(r'[^\w\s-]', '', title)[:30]
    return f"articles/{safe_title}_{hash_value}_{suffix}"

def generate_shorts(content: List[Dict]) -> str:
    """본문 내용에서 shorts 생성"""
    text_blocks = [
        block['content'] for block in content 
        if block['type'] == 'text'
    ]
    
    if text_blocks:
        return text_blocks[0][:100] + "..."
    return "본문 내용 없음"

def get_category_from_filename(filename: str) -> str:
    """파일명에서 카테고리 추출"""
    category_map = {
        'culture': 'CULTURE',
        'hobby': 'HOBBY',
        'investment': 'INVESTMENT',
        'real_estate': 'REAL_ESTATE',
        'travel': 'TRAVEL',
        'inheritance_gift': 'INHERITANCE_GIFT'
    }
    
    for key in category_map:
        if key in filename.lower():
            return category_map[key]
    return 'CULTURE'  # 기본값

@article_batch_bp.route('/api/articles', methods=['POST'])
def batch_insert():
    """processed 폴더의 모든 JSON 파일을 DB에 일괄 삽입"""
    try:
        processed_dir = Path("/app/app/utils/hanathenext_articles/processed")
        db = get_db()
        cursor = db.cursor()
        
        query = """
        INSERT INTO article (
            category,
            content,
            title,
            published_at,
            like_count,
            shorts,
            thumbnails3key,
            ttss3key
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        inserted_count = 0
        errors = []

        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                
                # 카테고리 추출
                category = get_category_from_filename(json_file.name)
                
                # content 배열만 추출
                content = article_data.get('content', [])
                
                title = article_data.get('title', '')
                published_at = datetime.now()
                shorts = generate_shorts(content)
                
                # S3 키 생성
                content_hash = hashlib.md5(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
                thumbnails3key = f"thumbnail/{content_hash}.jpg"
                ttss3key = f"tts/{content_hash}.mp3"
                
                cursor.execute(query, (
                    category,
                    json.dumps(content, ensure_ascii=False),  # content 배열만 JSON으로 저장
                    title,
                    published_at,
                    0,  # 초기 좋아요 수
                    shorts,
                    thumbnails3key,
                    ttss3key
                ))
                db.commit()
                
                inserted_count += 1
                
            except Exception as e:
                errors.append({
                    "file": json_file.name,
                    "error": str(e)
                })
                continue
        
        cursor.close()
        
        return jsonify({
            "success": True,
            "message": f"{inserted_count}개의 파일이 처리되었습니다.",
            "inserted_count": inserted_count,
            "errors": errors
        }), 200
        
    except Exception as e:
        if 'cursor' in locals():
            cursor.close()
        return jsonify({
            "success": False,
            "message": f"처리 중 오류가 발생했습니다: {str(e)}",
            "errors": []
        }), 500

@article_batch_bp.route('/api/articles/update-content', methods=['POST'])
def update_content():
    try:
        db = get_db()
        cursor = db.cursor()
        
        # 모든 article 조회
        select_query = "SELECT article_id, content FROM article"
        cursor.execute(select_query)
        articles = cursor.fetchall()
        
        # content 업데이트 쿼리
        update_query = "UPDATE article SET content = %s WHERE article_id = %s"
        
        updated_count = 0
        errors = []
        
        for article in articles:
            try:
                # 현재 content가 이미 문자열이므로 파싱 필요
                current_content = article['content']
                
                # 이미 중괄호로 감싸져 있는지 확인
                if not current_content.startswith('{'):
                    # 배열을 중괄호로 감싸기
                    new_content_str = '{' + current_content + '}'
                    
                    # 업데이트 실행
                    cursor.execute(update_query, (
                        new_content_str,
                        article['article_id']
                    ))
                    db.commit()
                    
                    updated_count += 1
                
            except Exception as e:
                errors.append({
                    "article_id": article['article_id'],
                    "error": str(e)
                })
                continue
        
        cursor.close()
        
        return jsonify({
            "success": True,
            "message": f"{updated_count}개의 article이 업데이트되었습니다.",
            "updated_count": updated_count,
            "errors": errors
        }), 200
        
    except Exception as e:
        if 'cursor' in locals():
            cursor.close()
        return jsonify({
            "success": False,
            "message": f"처리 중 오류가 발생했습니다: {str(e)}",
            "errors": []
        }), 500