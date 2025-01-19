import re
import torch
from typing import List, Dict
from transformers import BertModel, BertTokenizer
from logging import getLogger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from Korpora import Korpora

logger = getLogger(__name__)

class BertEmbedding:
    def __init__(self, model_path: str = './Bert'):
        self.model = BertModel.from_pretrained('klue/bert-base')
        self.tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        self.word_frequencies = {}  # 단어 빈도 저장
        self._load_korpora()
        logger.info("모델 및 Korpora 초기화 완료")

    def _load_korpora(self):
        try:
            corpus_file = "/root/Korpora/namuwikitext/namuwikitext_20200302.dev"
            logger.info(f"코퍼스 파일 로드 시작: {corpus_file}")
            
            total_words = 0
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    words = re.findall(r'[가-힣]+', line)
                    for word in words:
                        if len(word) >= 2:  # 2글자 이상만 저장
                            self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1
                            total_words += 1
                    
                    if i % 1000 == 0:
                        logger.info(f"처리 중: {i}번째 줄, 현재 {len(self.word_frequencies)}개 단어")

            logger.info(f"Korpora 처리 완료: 총 {len(self.word_frequencies)}개 단어, {total_words}개 토큰")
        except Exception as e:
            logger.error(f"Korpora 로드 중 오류: {str(e)}")
            self.word_frequencies = {}

    def get_difficult_words(self, text: str) -> List[Dict[str, str]]:
        try:
            tokens = self.tokenizer.tokenize(text)
            logger.info(f"텍스트 토큰화 완료 (토큰 수: {len(tokens)})")
            
            MAX_LENGTH = 450
            MIN_THRESHOLD = 0.2  # 최소 임계값 설정
            token_chunks = [tokens[i:i + MAX_LENGTH] for i in range(0, len(tokens), MAX_LENGTH)]
            
            all_difficult_words = []
            
            for chunk_tokens in token_chunks:
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                
                inputs = self.tokenizer(
                    chunk_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state.squeeze(0)
                    
                    # BERT 기반 복잡도 계산
                    token_norms = torch.norm(hidden_states, dim=1)
                    token_scores = (token_norms - token_norms.min()) / (token_norms.max() - token_norms.min())
                    
                    for token, score in zip(chunk_tokens, token_scores):
                        if (len(token) >= 2 and 
                            not token.startswith('##') and
                            re.search(r'[가-힣]{2,}', token)):
                            
                            # 원본 단어 복원 (##제거)
                            word = token.replace('##', '')
                            
                            # Korpora 빈도 정보 활용
                            freq = self.word_frequencies.get(word, 0)
                            freq_score = 1.0 / (1.0 + freq)  # 빈도가 낮을수록 높은 점수
                            
                            # 최종 점수 계산 (BERT 점수 70%, 빈도 점수 30%)
                            final_score = 0.7 * float(score) + 0.3 * freq_score
                            
                            if final_score > MIN_THRESHOLD:  # 최소 임계값 체크
                                all_difficult_words.append({
                                    'word': word,
                                    'difficulty': final_score
                                })
            
            # 중복 제거 및 정렬
            unique_words = {}
            for item in all_difficult_words:
                word = item['word']
                if word not in unique_words or item['difficulty'] > unique_words[word]['difficulty']:
                    unique_words[word] = item
            
            sorted_words = sorted(
                unique_words.values(),
                key=lambda x: x['difficulty'],
                reverse=True
            )
            
            # 최소 3개의 단어 선택 (단, 최소 임계값 이상인 경우만)
            if len(sorted_words) < 3:
                logger.info(f"임계값({MIN_THRESHOLD}) 이상의 단어가 3개 미만입니다: {len(sorted_words)}개")
            
            logger.info(f"어려운 단어 추출 완료 (단어 수: {len(sorted_words)})")
            if sorted_words:
                logger.info(f"상위 단어 샘플: {[w['word'] for w in sorted_words[:5]]}")
            
            return sorted_words[:min(3, len(sorted_words))] 

        except Exception as e:
            logger.error(f"어려운 단어 추출 중 오류: {str(e)}")
            return []

    def _tokenize_to_words(self, text: str) -> List[str]:
        """BERT 토크나이저로 단어 추출"""
        try:
            tokens = self.tokenizer.tokenize(text)
            words = []
            for token in tokens:
                if not token.startswith('##') and token not in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                    words.append(token)
            return list(set(words))
        except Exception as e:
            logger.error(f"토큰화 중 오류: {str(e)}")
            return []

    def _calculate_bert_complexity(self, word: str) -> float:
        """BERT 모델 기반 단어 복잡도 계산"""
        try:
            inputs = self.tokenizer(
                word,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            complexity_score = torch.sigmoid(logits[0][1]).item()
            return complexity_score
            
        except Exception as e:
            logger.error(f"단어 복잡도 계산 중 오류: {str(e)}")
            return 0.5

    def _calculate_difficulty_scores(self, embeddings):
        # 이 메서드는 난이도 점수 계산 로직을 구현해야 합니다.
        # 현재는 임시로 임베딩 값을 난이도 점수로 사용합니다.
        return embeddings.mean(axis=0).tolist()