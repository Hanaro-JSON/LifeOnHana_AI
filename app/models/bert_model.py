from transformers import AlbertModel, BertTokenizer
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BertEmbedding:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = AlbertModel.from_pretrained(model_path)
        # 모델 정보 로깅
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
                # 모델 출력 shape 확인
                logger.info(f"모델 출력 shape: {outputs.last_hidden_state.shape}")
                
                # CLS 토큰의 임베딩만 사용
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
                logger.info(f"최종 임베딩 shape: {embedding.shape}")
                
                return embedding.astype(np.float32)  # float32로 타입 변환
                
        except Exception as e:
            logger.error(f"임베딩 생성 중 에러: {str(e)}", exc_info=True)
            raise e 