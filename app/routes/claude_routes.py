from flask import Blueprint, request, jsonify
import anthropic
from sqlalchemy import create_engine, text
import os
import json
import re

bp = Blueprint('claude', __name__)

# 클로드 API 설정
API_KEY = os.getenv("CLAUDE_API_KEY", "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")
client = anthropic.Anthropic(api_key=API_KEY)

# 데이터베이스 연결 설정
DB_ENDPOINT = "lifeonhana.cxq2u4wk2434.ap-northeast-2.rds.amazonaws.com"
DB_PORT = 3306
DB_USERNAME = "admin"
DB_PASSWORD = "LifeOnHana1!"
DB_NAME = "lifeonhana_serverDB"

DATABASE_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_ENDPOINT}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


@bp.route('/related_products', methods=['POST'])
def related_products():
    content = request.json.get('content')  # JSON 배열 가져오기
    if not isinstance(content, list):  # content가 배열인지 확인
        return jsonify({"error": "content must be a list"}), 400

    try:
        # `type`이 "TEXT"인 항목만 필터링하여 문자열로 결합
        user_prompt = " ".join(
            part['content'] for part in content if part.get('type') == 'TEXT'
        )

        if not user_prompt.strip():  # 필터링 결과가 비어있다면 에러 처리
            return jsonify({"error": "No valid TEXT content found in the input"}), 400

        # 데이터베이스에서 모든 상품 조회
        query = text("""
            SELECT product_id, name, description, category, link
            FROM product_test
        """)
        with engine.connect() as connection:
            products = connection.execute(query).mappings().fetchall()

        # 상품 리스트를 JSON 문자열로 변환
        product_data = [
            {"id": product["product_id"], "name": product["name"], "description": product["description"]}
            for product in products
        ]
        product_json = json.dumps(product_data, ensure_ascii=False)

        # 클로드 API 호출
        response = client.completions.create(
            model="claude-2.0",
            max_tokens_to_sample=4096,
            prompt=f"""
                {anthropic.HUMAN_PROMPT}
                The following is an article content about inheritance and gifting:
                "{user_prompt}"
                
                Below are several financial and life-related products. Please analyze their relevance to the article content and score each product from 0 to 100:
                
                Products:
                {product_json}

                Provide the **top 2 most relevant products** in JSON format, including their IDs and relevance scores:
                [{{"id": 101, "score": 95}}, {{"id": 102, "score": 90}}]

                The scoring criteria should prioritize the match between the product description and the main topics in the article content.
                {anthropic.AI_PROMPT}
            """
        )
        analysis_result = response.completion.strip()

        try:
            # 정규 표현식을 사용해 JSON 배열 추출
            match = re.search(r'\[.*?\]', analysis_result, re.DOTALL)
            if match:
                json_data = match.group(0)  # JSON 데이터 추출
                top_products = json.loads(json_data)  # JSON 파싱
            else:
                raise ValueError("No JSON data found in Claude API response")

        except (ValueError, json.JSONDecodeError):
            return jsonify({
                "error": "Claude API did not return valid JSON",
                "raw_response": analysis_result
            }), 500

        # 상위 2개의 상품 데이터 생성
        selected_products = []
        for product in products:
            for top_product in top_products:
                if product["product_id"] == top_product["id"]:
                    selected_products.append({
                        "product_id": product["product_id"],
                        "name": product["name"],
                        "category": product["category"],
                        "link": product["link"],
                        "score": top_product["score"]
                    })

        return jsonify({"products": selected_products}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
