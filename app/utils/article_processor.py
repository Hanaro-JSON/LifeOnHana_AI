import os
import json
import logging
import requests
from typing import List, Dict
from app.models.bert_model import BertEmbedding

logger = logging.getLogger(__name__)

class ArticleProcessor:
    def __init__(self, bert_model):
        self.bert = bert_model
        self.api_url = "http://localhost:5000/ask_claude"  # 로컬 API 엔드포인트

    def _get_batch_descriptions(self, words: List[Dict], batch_size: int = 5) -> Dict[str, str]:
        """단어 설명을 배치로 가져오기"""
        descriptions = {}
        
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            word_list = [w['word'] for w in batch_words]
            
            prompt = (
                f"다음 단어들의 의미를 간단히 설명해주세요. "
                f"각 단어는 한 문장으로 설명하고 '단어: 설명' 형식으로 작성해주세요.\n\n"
                f"단어 목록: {', '.join(word_list)}"
            )
            
            try:
                response = requests.post(
                    self.api_url,
                    json={"prompt": prompt}
                )
                response.raise_for_status()
                response_data = response.json()
                batch_descriptions = self._parse_claude_response(response_data["response"])
                descriptions.update(batch_descriptions)
                
            except Exception as e:
                logger.error(f"Error getting descriptions for batch: {str(e)}")
                
        return descriptions

    def _parse_claude_response(self, response: str) -> Dict[str, str]:
        """Claude 응답 파싱"""
        descriptions = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                word, desc = line.split(':', 1)
                descriptions[word.strip()] = desc.strip()
                
        return descriptions

    def process_articles_from_folder(self, folder_path: str):
        """폴더 내 모든 기사 처리"""
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    
                    # 각 텍스트 블록에서 어려운 단어 추출
                    for block in article['content']:
                        if block['type'] == 'text':
                            text = block['content']
                            difficult_words = self.bert.get_difficult_words(text)
                            
                            if difficult_words:
                                # 어려운 단어 설명 가져오기
                                descriptions = self._get_batch_descriptions(difficult_words)
                                
                                # 설명이 있는 단어만 저장
                                block['difficult_words'] = [
                                    {
                                        'word': word['word'],
                                        'description': descriptions.get(word['word'], ''),
                                        'difficulty_score': word['score']
                                    }
                                    for word in difficult_words
                                    if word['word'] in descriptions
                                ]
                    
                    # 처리된 기사 저장
                    output_path = os.path.join(folder_path, f"processed_{filename}")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(article, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    logger.error(f"Error processing article {filename}: {str(e)}")

# 사용 예시
if __name__ == "__main__":
    bert_model = BertEmbedding(os.getenv('MODEL_PATH', './Bert'))
    processor = ArticleProcessor(bert_model)
    processor.process_articles_from_folder("path/to/articles") 