from flask import Blueprint, request, jsonify
import anthropic
from sqlalchemy import create_engine, text
import os
import json
import re
from config.settings import MYSQL_CONFIG


bp = Blueprint('claude', __name__)

@bp.route("/")  # 기본 경로 추가
def home():
    return "Welcome to the LifeOnHana AI Service!"

# 클로드 API 설정
API_KEY = os.getenv("CLAUDE_API_KEY", "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")
client = anthropic.Anthropic(api_key=API_KEY)

DATABASE_URL = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

@bp.route('/related_products', methods=['POST'])
def related_products():
    try:
        content = request.json.get('content')  # JSON 배열 가져오기
        print("✅ Received Content:", content, flush=True)

        if not isinstance(content, list):  # content가 배열인지 확인
            return jsonify({"error": "content must be a list"}), 400

        # `type`이 "TEXT"인 항목만 필터링하여 문자열로 결합
        user_prompt = " ".join(
            part['content'] for part in content if part.get('type') == 'text'
        )

        print("📝 User Prompt:", user_prompt, flush=True)

        if not user_prompt.strip():  # 필터링 결과가 비어있다면 에러 처리
            return jsonify({"error": "No valid TEXT content found in the input"}), 400

        # 데이터베이스에서 모든 상품 조회
        query = text("""
            SELECT product_id, name, description, category, link
            FROM product
        """)
        with engine.connect() as connection:
            products = connection.execute(query).mappings().fetchall()

        print("📦 Retrieved Products:", products, flush=True)

        # 상품 리스트를 JSON 문자열로 변환
        product_data = [
            {"id": product["product_id"], "name": product["name"], "description": product["description"]}
            for product in products
        ]
        product_json = json.dumps(product_data, ensure_ascii=False)

        # 클로드 API 호출
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"Analyze the relevance of the following products to the article: {user_prompt}. Products: {product_json}. Return top 2 relevant products in JSON format."}]
        )

        print("📦 Claude API Response:", response, flush=True)

        # 응답 파싱
        analysis_result = " ".join(
            item.text.strip() for item in response.content if hasattr(item, 'text')
        ) if isinstance(response.content, list) else response.content.strip()

        print("✅ Final Analysis Result:", analysis_result, flush=True)

        # JSON 배열 추출
        match = re.search(r'\[.*?\]', analysis_result, re.DOTALL)
        if not match:
            raise ValueError("No JSON data found in Claude API response")

        json_data = match.group(0)
        top_products = json.loads(json_data)

        # 상품 ID로 매칭하여 관련 상품 반환
        selected_products = [
            {
                "product_id": product["product_id"],
                "name": product["name"],
                "category": product["category"],
                "link": product["link"],
                "description": product["description"]
            }
            for product in products
            if any(item.get("id") == product["product_id"] for item in top_products)
        ]

        return jsonify({"products": selected_products[:2]}), 200

    except Exception as e:
        print("❌ Error:", str(e), flush=True)
        return jsonify({"error": str(e)}), 500


@bp.route('/recommend_loan_products', methods=['POST'])
def recommend_loan_products():
    try:
        data = request.json
        print("✅ Received Data:", data, flush=True)

        reason = data.get('reason')
        amount = data.get('amount')
        user_data = data.get('userData', {})
        products = data.get('products', [])

        if not reason or not isinstance(reason, str) or not reason.strip():
            return jsonify({"error": "'reason' must be a non-empty string"}), 400
        if not amount or not isinstance(amount, (int, float)):
            return jsonify({"error": "'amount' must be a valid number"}), 400
        if not isinstance(products, list) or not products:
            return jsonify({"error": "'products' must be a non-empty list"}), 400

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"Recommend top 5 loan products based on the user's reason: {reason}, amount: {amount}, and user data: {json.dumps(user_data)}. Products: {json.dumps(products, ensure_ascii=False)}."}]
        )

        print("📦 Claude API Response:", response, flush=True)

        analysis_result = " ".join(
            item.text.strip() for item in response.content if hasattr(item, 'text')
        ) if isinstance(response.content, list) else response.content.strip()

        print("✅ Final Analysis Result:", analysis_result, flush=True)

        match = re.search(r'\[.*?\]', analysis_result, re.DOTALL)
        if not match:
            raise ValueError("Invalid JSON format from Claude API")

        json_data = match.group(0)
        top_products = json.loads(json_data)

        selected_products = [
            {
                **product,
                "score": next((item["score"] for item in top_products if item["id"] == product["id"]), 0)
            }
            for product in products if product["id"] in [item["id"] for item in top_products]
        ]

        selected_products.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"products": selected_products[:5]}), 200

    except Exception as e:
        print("❌ Error:", str(e), flush=True)
        return jsonify({"error": str(e)}), 500


# @bp.route("/")  # 기본 경로 추가
# def home():
#     return "Welcome to the LifeOnHana AI Service!"

# # 클로드 API 설정
# API_KEY = os.getenv("CLAUDE_API_KEY", "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")
# client = anthropic.Anthropic(api_key=API_KEY)

# DATABASE_URL = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
# engine = create_engine(DATABASE_URL)

# @bp.route('/related_products', methods=['POST'])
# def related_products():
#     content = request.json.get('content')  # JSON 배열 가져오기
#     if not isinstance(content, list):  # content가 배열인지 확인
#         return jsonify({"error": "content must be a list"}), 400

#     try:
#         # `type`이 "TEXT"인 항목만 필터링하여 문자열로 결합
#         user_prompt = " ".join(
#             part['content'] for part in content if part.get('type') == 'text'
#         )

#         if not user_prompt.strip():  # 필터링 결과가 비어있다면 에러 처리
#             return jsonify({"error": "No valid TEXT content found in the input"}), 400

#         # 데이터베이스에서 모든 상품 조회
#         query = text("""
#             SELECT product_id, name, description, category, link
#             FROM product
#         """)
#         with engine.connect() as connection:
#             products = connection.execute(query).mappings().fetchall()

#         # 상품 리스트를 JSON 문자열로 변환
#         product_data = [
#             {"id": product["product_id"], "name": product["name"], "description": product["description"]}
#             for product in products
#         ]
#         product_json = json.dumps(product_data, ensure_ascii=False)

#         # 클로드 API 호출
#         response = client.completions.create(
#             model="claude-2.0",
#             max_tokens_to_sample=2048,
#             prompt=f"""
#                 {anthropic.HUMAN_PROMPT}
#                 The following is an article content about inheritance and gifting:
#                 "{user_prompt}"
                
#                 Below are several financial and life-related products. Please analyze their relevance to the article content and score each product from 0 to 100:
                
#                 Products:
#                 {product_json}

#                 Provide the **top 2 most relevant products** in JSON format, including their IDs and relevance scores:
#                 [{{"id": 101, "score": 95}}, {{"id": 102, "score": 90}}]

#                 The scoring criteria should prioritize the match between the product description and the main topics in the article content.
#                 {anthropic.AI_PROMPT}
#             """
#         )
#         analysis_result = response.completion.strip()

#         try:
#             # 정규 표현식을 사용해 JSON 배열 추출
#             match = re.search(r'\[.*?\]', analysis_result, re.DOTALL)
#             if match:
#                 json_data = match.group(0)  # JSON 데이터 추출
#                 top_products = json.loads(json_data)  # JSON 파싱
#             else:
#                 raise ValueError("No JSON data found in Claude API response")

#         except (ValueError, json.JSONDecodeError):
#             return jsonify({
#                 "error": "Claude API did not return valid JSON",
#                 "raw_response": analysis_result
#             }), 500

#         # 상위 2개의 상품 데이터 생성
#         selected_products = []
#         for product in products:
#             for top_product in top_products:
#                 if product["product_id"] == top_product["id"]:
#                     selected_products.append({
#                         "product_id": product["product_id"],
#                         "name": product["name"],
#                         "category": product["category"],
#                         "link": product["link"],
#                         "score": top_product["score"]
#                     })

#         return jsonify({"products": selected_products}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @bp.route('/recommend_loan_products', methods=['POST'])
# def recommend_loan_products():
#     data = request.json

#     # 요청 데이터 검증
#     reason = data.get('reason')
#     amount = data.get('amount')
#     user_data = data.get('userData', {})
#     products = data.get('products', [])

#     if not reason or not isinstance(reason, str) or not reason.strip():
#         return jsonify({"error": "'reason' must be a non-empty string"}), 400
#     if not amount or not isinstance(amount, (int, float)):
#         return jsonify({"error": "'amount' must be a valid number"}), 400
#     if not isinstance(products, list) or not products:
#         return jsonify({"error": "'products' must be a non-empty list"}), 400

#     # userData 기본값 설정
#     user_data = {
#         "deposit_amount": user_data.get("deposit_amount", 0),
#         "loan_amount": user_data.get("loan_amount", 0),
#         "real_estate_amount": user_data.get("real_estate_amount", 0),
#         "total_asset": user_data.get("total_asset", 0),
#     }

#     # Anthropic API 호출
#     try:
#         response = client.completions.create(
#             model="claude-2.0",
#             max_tokens_to_sample=2048,
#             prompt=f"""
#                 {anthropic.HUMAN_PROMPT}
#                 The user is requesting loan products for the reason: "{reason}", with a requested amount of {amount}.

#                 The user's financial summary is:
#                 - Deposit Amount: {user_data['deposit_amount']}
#                 - Loan Amount: {user_data['loan_amount']}
#                 - Real Estate Amount: {user_data['real_estate_amount']}
#                 - Total Asset: {user_data['total_asset']}

#                 Available loan products:
#                 {json.dumps(products, ensure_ascii=False)}

#                 Please return **the top 5 most relevant products** in JSON format:
#                 [{{"id": 101, "score": 95}}, {{"id": 102, "score": 90}}]
#                 {anthropic.AI_PROMPT}
#             """
#         )
#         analysis_result = response.completion.strip()

#         # JSON 배열 추출
#         match = re.search(r'\[.*?\]', analysis_result, re.DOTALL)
#         if not match:
#             raise ValueError("Invalid JSON format from Claude API")

#         json_data = match.group(0)
#         top_products = json.loads(json_data)

#         # 결과 생성
#         selected_products = [
#             {
#                 **product,
#                 "score": next((item["score"] for item in top_products if item["id"] == product["id"]), 0)
#             } for product in products if product["id"] in [item["id"] for item in top_products]
#         ]

#         selected_products.sort(key=lambda x: x["score"], reverse=True)
#         return jsonify({"products": selected_products[:5]}), 200

#     except json.JSONDecodeError as jde:
#         return jsonify({"error": "Failed to decode JSON response", "details": str(jde)}), 500
#     except ValueError as ve:
#         return jsonify({"error": "Invalid response format", "details": str(ve)}), 500
#     except Exception as e:
#         return jsonify({"error": "Unexpected error occurred", "details": str(e)}), 500


# @bp.route('/effect', methods=['POST'])
# def recommend_effect():
#     try:
#         data = request.json
#         product = data.get("product", {})
#         article_shorts = data.get("articleShorts", "").strip()
#         user_data = data.get("userData", {})
        
#         # 요청 데이터 검증
#         if not product or not article_shorts:
#             return jsonify({"error": "Product and articleShorts are required"}), 400

#         category = product.get("category")
#         if category not in ["LOAN", "SAVINGS", "LIFE"]:
#             return jsonify({"error": "Invalid product category"}), 400

#         if category in ["LOAN", "SAVINGS"]:
#             # 금융 상품
#             prompt = f"""
#             {anthropic.HUMAN_PROMPT}
#             The user is considering a financial product. Below is the context:
            
#             - Article Content: "{article_shorts}"
#             - Product Name: "{product.get('name', 'N/A')}"
#             - Description: "{product.get('description', 'N/A')}"
#             - Interest Rate: Basic: {product.get('basic_interest_rate', 'N/A')}, Max: {product.get('max_interest_rate', 'N/A')}
#             - Amount Range: Min: {product.get('min_amount', 'N/A')}, Max: {product.get('max_amount', 'N/A')}
            
#             User's Financial Data:
#             - Total Asset: {user_data.get('total_asset', 0)}
#             - Deposit Amount: {user_data.get('deposit_amount', 0)}
#             - Savings Amount: {user_data.get('savings_amount', 0)}
#             - Loan Amount: {user_data.get('loan_amount', 0)}
            
#             Please generate a personalized recommendation for the user regarding this financial product without explicitly mentioning phrases like "Based on the context provided" or "Here is a personalized recommendation"
#             {anthropic.AI_PROMPT}
#             """
#         else:
#             # 라이프 상품
#             recent_histories = user_data.get("recent_histories", [])
#             formatted_histories = "\n".join(
#                 f"- {history.get('category', 'N/A')}: {history.get('description', 'N/A')} (Amount: {history.get('amount', 'N/A')})"
#                 for history in recent_histories
#             )
#             prompt = f"""
#             {anthropic.HUMAN_PROMPT}
#             The following is a lifestyle product recommendation context:

#             - Article Content: "{article_shorts}"
#             - Product Name: "{product.get('name', 'N/A')}"
#             - Description: "{product.get('description', 'N/A')}"

#             User's Recent Activities:
#             {formatted_histories}

#             Generate a concise and engaging personalized recommendation for the user, without explicitly mentioning phrases like "Based on the context provided" or "Here is a personalized recommendation". Focus directly on the user's context and why this product is a good fit.
#             {anthropic.AI_PROMPT}
#             """

#         response = client.completions.create(
#             model="claude-2.0",
#             max_tokens_to_sample=4096,
#             prompt=prompt.strip()
#         )

#         analysis_result = response.completion.strip()

#         return jsonify({
#             "analysisResult": analysis_result,
#             "productLink": product.get("link", "N/A")
#         }), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@bp.route('/effect', methods=['POST'])
def recommend_effect():
    try:
        data = request.json
        print("✅ Received Data:", data, flush=True)

        product = data.get("product", {})
        article_shorts = data.get("articleShorts", "")

        if isinstance(article_shorts, list):
            article_shorts = " ".join(
                str(item).strip() for item in article_shorts if isinstance(item, str) and item.strip()
            )
        elif isinstance(article_shorts, str):
            article_shorts = article_shorts.strip()
        else:
            article_shorts = str(article_shorts).strip()

        user_data = data.get("userData", {})

        if not product or not article_shorts:
            return jsonify({"error": "Product and articleShorts are required"}), 400

        category = product.get("category")
        if category not in ["LOAN", "SAVINGS", "LIFE"]:
            return jsonify({"error": "Invalid product category"}), 400

        if category in ["LOAN", "SAVINGS"]:
            prompt = f"""
            The user is considering a financial product. Below is the context:

            - Article Content: "{article_shorts}"
            - Product Name: "{product.get('name', 'N/A')}"
            - Description: "{product.get('description', 'N/A')}"
            - Interest Rate: Basic: {product.get('basic_interest_rate', 'N/A')}, Max: {product.get('max_interest_rate', 'N/A')}"
            - Amount Range: Min: {product.get('min_amount', 'N/A')}, Max: {product.get('max_amount', 'N/A')}"

            User's Financial Data:
            - Total Asset: {user_data.get('total_asset', 0)}
            - Deposit Amount: {user_data.get('deposit_amount', 0)}
            - Savings Amount: {user_data.get('savings_amount', 0)}
            - Loan Amount: {user_data.get('loan_amount', 0)}

            Please generate a personalized recommendation.
            """
        else:
            recent_histories = user_data.get("recent_histories", [])
            formatted_histories = "\n".join(
                f"- {history.get('category', 'N/A')}: {history.get('description', 'N/A')} (Amount: {history.get('amount', 'N/A')})"
                for history in recent_histories
            )
            prompt = f"""
            The following is a lifestyle product recommendation context:

            - Article Content: "{article_shorts}"
            - Product Name: "{product.get('name', 'N/A')}"
            - Description: "{product.get('description', 'N/A')}"
            - User's Recent Activities:
            {formatted_histories}

            Generate a concise and engaging personalized recommendation.
            """

        print("📝 Generated Prompt:", prompt.strip(), flush=True)

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt.strip()}]
        )

        print("📦 Claude API Response:", response, flush=True)

        # 📌 API 응답에서 `text` 필드 추출
        if isinstance(response.content, list):
            analysis_result = " ".join(
                item.text.strip() for item in response.content if hasattr(item, 'text')
            )
        elif isinstance(response.content, str):
            analysis_result = response.content.strip()
        else:
            analysis_result = str(response.content).strip()

        print("✅ Final Analysis Result:", analysis_result, flush=True)

        return jsonify({
            "analysisResult": analysis_result,
            "productLink": product.get("link", "N/A")
        }), 200

    except Exception as e:
        print("❌ Error:", str(e), flush=True)
        return jsonify({"error": str(e)}), 500
