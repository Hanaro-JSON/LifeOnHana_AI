from flask import Blueprint, request, jsonify
import anthropic

bp = Blueprint('claude', __name__)

client = anthropic.Anthropic(
    api_key="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"  # API 키는 환경변수로 이동하는 것이 좋습니다
)

@bp.route('/ask_claude', methods=['POST'])
def ask_claude():
    user_prompt = request.json.get('prompt')
    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response = client.completions.create(
            model="claude-2.0",
            max_tokens_to_sample=4096,  
            prompt=f"{anthropic.HUMAN_PROMPT} {user_prompt}{anthropic.AI_PROMPT}"
        )
        return jsonify({"response": response.completion}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500 