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

    def _get_batch_descriptions(self, words: List[Dict], batch_size: int = 5) -> Dict[str, str]:
        """단어 설명을 배치로 가져오기"""
        descriptions = {}
        
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            word_list = [w['word'] for w in batch_words]
            
            try:
                # logger.info(f"Claude API 요청 - 단어: {word_list}")
                
                message = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": f"다음 단어들의 의미를 간단히 설명해주세요. 각 단어는 한 문장으로만 설명하고, '단어: 설명' 형식으로 작성해주세요.\n\n단어 목록: {', '.join(word_list)}"
                    }]
                )
                
                if message.content:
                    content = message.content[0].text
                    # logger.info(f"Claude 응답: {content}")
                    
                    # 응답 파싱
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if ':' in line:
                            word, desc = line.split(':', 1)
                            word = word.strip()
                            desc = desc.strip()
                            if word and desc:
                                descriptions[word] = desc
                                # logger.info(f"단어 설명 추가: {word} -> {desc}")
                
            except Exception as e:
                # logger.error(f"API 요청/처리 중 오류: {str(e)}")
                continue
        
        return descriptions 