import anthropic
import logging
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

class ArticleProcessor:
    def __init__(self, bert_model):
        self.bert_model = bert_model
        self.client = anthropic.Anthropic(
            api_key="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"
        )
        # logger.info("Claude API 클라이언트 초기화 완료")

    def _get_batch_descriptions(self, words: List[Dict]) -> Dict[str, str]:
        descriptions = {}
        
        try:
            message = self.client.messages.create(
                model="claude-2.1",  # Claude 2 모델 사용
                max_tokens=300,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": f"다음 단어들의 의미를 최대한 간단히 설명해주세요 (각 단어당 10단어 이내):\n{', '.join(w['word'] for w in words)}"
                }]
            )
            
            if message.content:
                content = message.content[0].text
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                for line in lines:
                    if ':' in line:
                        word, desc = line.split(':', 1)
                        word = word.strip()
                        desc = desc.strip()
                        if word and desc:
                            descriptions[word] = desc
        
        except Exception as e:
            logger.error(f"API 요청/처리 중 오류: {str(e)}")
        
        return descriptions 