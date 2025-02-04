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
API_KEY = os.getenv("CLAUDE_API_KEY")

if not API_KEY:
    print("❌ CLAUDE_API_KEY가 설정되지 않았습니다.", flush=True)
else:
    print(f"✅ API 키 로드 완료: {API_KEY[:8]}***", flush=True)
    
client = anthropic.Anthropic(api_key=API_KEY)

DATABASE_URL = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
engine = create_engine(DATABASE_URL)

@bp.route('/related_products', methods=['POST'])
def related_products():
    try:
        content = request.json.get('content')
        print("✅ Received Content:", content, flush=True)

        if not isinstance(content, str):
            return jsonify({"error": "content must be a string"}), 400

        user_prompt = content.strip()
        print("📝 User Prompt:", user_prompt, flush=True)

        if not user_prompt:
            return jsonify({"error": "No valid content provided"}), 400

        query = text("""
            SELECT product_id, name, description, category, link
            FROM product
        """)
        with engine.connect() as connection:
            products = connection.execute(query).mappings().fetchall()

        print("📦 Retrieved Products:", products, flush=True)

        product_data = [
            {"id": product["product_id"], "name": product["name"], "description": product["description"]}
            for product in products
        ]
        product_json = json.dumps(product_data, ensure_ascii=False)

            
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""
                Analyze the relevance of the following products to the article:
                "{user_prompt}"

                Products:
                {product_json}

                Return ONLY the **top 2 most relevant products** in strict JSON format with IDs:
                [
                    {{"id": 101}},
                    {{"id": 102}}
                ]
                ⚠️ Do NOT provide any additional explanation or text. ONLY return the JSON array as shown above.
                """
            }]
        )

        print("📦 Claude API Response:", response, flush=True)

        analysis_result = (
            " ".join(item.text.strip() for item in response.content if hasattr(item, 'text'))
            if isinstance(response.content, list)
            else response.content.strip()
        )

        print("✅ Final Analysis Result:", analysis_result, flush=True)

        match = re.search(r'\[.*?\]', analysis_result, re.DOTALL)
        if not match:
            raise ValueError("No JSON data found in Claude API response")

        json_data = match.group(0)
        
        try:
            top_products = json.loads(json_data)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Error: {e}")
            print("🚩 Problematic JSON:", json_data)
            raise

        if len(top_products) < 2:
            return jsonify({"products": []}), 200

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

        user_data = {
            "deposit_amount": user_data.get("deposit_amount", 0),
            "loan_amount": user_data.get("loan_amount", 0),
            "real_estate_amount": user_data.get("real_estate_amount", 0),
            "total_asset": user_data.get("total_asset", 0),
        }

        prompt = f"""
        The user is requesting loan products for the reason: "{reason}", with a requested amount of {amount}.

        The user's financial summary is:
        - Deposit Amount: {user_data['deposit_amount']}
        - Loan Amount: {user_data['loan_amount']}
        - Real Estate Amount: {user_data['real_estate_amount']}
        - Total Asset: {user_data['total_asset']}

        Available loan products:
        {json.dumps(products, ensure_ascii=False)}

        Please return the **top 5 most relevant products** in **valid JSON format** like below:
        [
            {{"id": 101, "score": 95}},
            {{"id": 102, "score": 90}}
        ]
        Only return the JSON array, without any additional text or explanation.
        """

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt.strip()}]
        )

        analysis_result = " ".join(
            item.text.strip() for item in response.content if hasattr(item, 'text')
        ) if isinstance(response.content, list) else response.content.strip()

        print("📦 Claude API Response:", analysis_result, flush=True)

        match = re.search(r'\[.*?\]', analysis_result, re.DOTALL)
        if not match:
            raise ValueError("Invalid JSON format from Claude API")

        json_data = match.group(0)
        top_products = json.loads(json_data)

        selected_products = [
            {
                **product,
                "score": next((item["score"] for item in top_products if item["id"] == product["id"]), 0)
            } for product in products if product["id"] in [item["id"] for item in top_products]
        ]

        selected_products.sort(key=lambda x: x["score"], reverse=True)

        return jsonify({"products": selected_products[:5]}), 200

    except json.JSONDecodeError as jde:
        return jsonify({"error": "Failed to decode JSON response", "details": str(jde)}), 500
    except ValueError as ve:
        return jsonify({"error": "Invalid response format", "details": str(ve)}), 500
    except Exception as e:
        return jsonify({"error": "Unexpected error occurred", "details": str(e)}), 500

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

            Provide ONLY the personalized recommendation in one concise paragraph. 
            DO NOT include any background explanation or introduction like "Based on the user's data". 
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

            Provide ONLY the personalized recommendation in one concise paragraph. 
            DO NOT include any background explanation or introduction like "Based on the user's activities". 
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
