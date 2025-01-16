from transformers import AlbertModel, BertTokenizer
import torch
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class BertEmbedding:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = AlbertModel.from_pretrained(model_path)
        self.model.eval()
        self.difficulty_threshold = 0.65
        logger.info(f"모델 설정: {self.model.config}")
        
    def get_embedding(self, text):
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
                return embedding.astype(np.float32)
                
        except Exception as e:
            logger.error(f"임베딩 생성 중 에러: {str(e)}", exc_info=True)
            raise e 

    def get_difficult_words(self, text: str) -> List[Dict]:
        """문장에서 어려운 단어 추출"""
        # BERT 토크나이저로 단어 분리
        tokens = self.tokenizer.tokenize(text)
        
        # 의미있는 토큰만 선택 (##이 없는 토큰)
        words = [token for token in tokens if not token.startswith('##') and len(token) > 1]
        
        if not words:
            return []

        # 중복 제거
        words = list(set(words))

        # BERT 임베딩 계산
        inputs = self.tokenizer(words, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            word_embeddings = outputs.last_hidden_state

        # 각 단어의 난이도 분석
        difficult_words = []
        for idx, word in enumerate(words):
            # 임베딩 기반 난이도 계산
            embedding = word_embeddings[idx].mean(dim=0)
            magnitude = torch.norm(embedding).item()
            variance = torch.var(embedding).item()
            
            # 난이도 점수 계산
            difficulty_score = (
                0.7 * (magnitude / 10) +  # 벡터 크기
                0.3 * (variance / 5)      # 벡터 분산
            )
            final_score = min(max(difficulty_score, 0.0), 1.0)
            
            # 난이도 기준치를 넘는 경우만 추가
            if final_score >= self.difficulty_threshold:
                difficult_words.append({
                    'word': word,
                    'score': final_score
                })
        
        return sorted(difficult_words, key=lambda x: x['score'], reverse=True)