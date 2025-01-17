from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import List, Dict
from korpora import Korpora, KoreanCorpus

logger = logging.getLogger(__name__)

class BertEmbedding:
    def __init__(self, model_path):
        try:
            # AlbertForMaskedLM 대신 BertForSequenceClassification 사용
            self.tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
            self.model = BertForSequenceClassification.from_pretrained(
                'klue/bert-base',
                num_labels=2  # 어려움/쉬움 두 가지 클래스
            )
            self.model.eval()
            logger.info("BERT 모델 초기화 완료")
            
            # Korpora 데이터 로드 (예: 모던 코퍼스)
            self.corpus = Korpora.load('korean_modern')
            self.word_frequencies = self._calculate_word_frequencies()
        except Exception as e:
            logger.error(f"모델 초기화 중 오류: {str(e)}")
            raise e

    def get_embedding(self, text):
        """텍스트 임베딩 생성 (기존 기능 유지)"""
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
                embedding = outputs.logits[0].numpy()
                return embedding.astype(np.float32)
                
        except Exception as e:
            logger.error(f"임베딩 생성 중 에러: {str(e)}", exc_info=True)
            raise e

    def preprocess_text(self, text):
        """텍스트 전처리"""
        # 특수문자 제거 (한글, 영문, 숫자 유지)
        text = re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)
        return text

    def get_difficult_words(self, text: str, threshold: float = 0.7) -> List[Dict]:
        """어려운 단어 추출 - BERT 임베딩과 빈도 정보 결합"""
        words = self._tokenize_to_words(text)
        difficult_words = []
        
        for word in words:
            if len(word) < 2:  # 한 글자 단어는 제외
                continue
                
            # BERT 기반 복잡도 점수 (기존 로직)
            bert_score = self._calculate_bert_complexity(word)
            
            # 빈도 기반 점수
            frequency = self.word_frequencies.get(word, 0)
            frequency_score = 1.0 - min(frequency * 1000, 1.0)  # 빈도가 낮을수록 높은 점수
            
            # 최종 난이도 점수 계산 (BERT와 빈도 결합)
            difficulty_score = (bert_score * 0.6) + (frequency_score * 0.4)
            
            if difficulty_score > threshold:
                difficult_words.append({
                    'word': word,
                    'difficulty_score': difficulty_score,
                    'frequency': frequency,
                    'bert_score': bert_score
                })
        
        return difficult_words

    def _tokenize_to_words(self, text: str) -> List[str]:
        try:
            # mecab으로 형태소 분석
            words = []
            for token in self.tokenizer.tokenize(text):
                # 특수 토큰 제외
                if token.startswith('##') or token in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                    continue
                words.append(token)
            return list(set(words))  # 중복 제거
        except Exception as e:
            logger.error(f"토큰화 중 오류: {str(e)}")
            return []

    def _calculate_bert_complexity(self, word: str) -> float:
        """BERT 모델을 사용한 단어 복잡도 계산"""
        # 기존 BERT 기반 복잡도 계산 로직
        return complexity_score

    def _calculate_word_frequencies(self):
        """말뭉치에서 단어 빈도 계산"""
        frequencies = {}
        total_words = 0
        
        for document in self.corpus.documents:
            for word in document.words:
                frequencies[word] = frequencies.get(word, 0) + 1
                total_words += 1
                
        # 빈도를 비율로 변환
        for word in frequencies:
            frequencies[word] = frequencies[word] / total_words
            
        return frequencies