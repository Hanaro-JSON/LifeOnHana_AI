import re
import torch
from typing import List, Dict
from transformers import BertModel, BertTokenizer
from logging import getLogger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from konlpy.tag import Mecab
from Korpora import Korpora
import re

logger = getLogger(__name__)

class BertEmbedding:
    def __init__(self, model_path: str = './Bert'):
        self.model = BertModel.from_pretrained('klue/bert-base')
        self.tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        self.word_frequencies = {}
        self._load_korpora()
        
        # 한국어 사전 경로 지정
        # self.mecab = Mecab('/usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ko-dic')
        # logger.info("모델 및 Korpora 초기화 완료")

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
            # 불용어 목록 확장
            stop_words = {
                # 조사/어미
                '으며', '이며', '하며', '되며', '즐기', '어울리',
                '어야', '어도', '으면', '이면', '아요', '어요',
                '에요', '예요', '으니', '이니', '는데', '은데',
                '으로', '에서', '부터', '까지', '마다', '처럼',
                
                # 동사/형용사 어간
                '되다', '하다', '있다', '없다', '이다', '아니다',
                '그렇다', '이렇다', '저렇다', '스럽다', '답다',
                
                # 대명사
                '그것', '이것', '저것', '그런', '이런', '저런',
                
                # 접속사
                '그리고', '하지만', '또는', '또한', '그러나',
                
                # 부사
                '매우', '너무', '아주', '정말', '거의', '바로',
                
                # 보조 용언
                '되어', '하여', '받아', '주어', '가지'
            }
            
            # nouns()를 사용하여 명사만 추출
            tokens = self.mecab.nouns(text)
            logger.info(f"텍스트 토큰화 완료 (토큰 수: {len(tokens)})")
            
            MAX_LENGTH = 450
            MIN_THRESHOLD = 0.2
            
            # 불용어 및 짧은 토큰 필터링
            filtered_tokens = [
                token for token in tokens 
                if len(token) >= 2 
                and token not in stop_words 
                and re.search(r'[가-힣]{2,}', token)
            ]
            
            token_chunks = [filtered_tokens[i:i + MAX_LENGTH] 
                           for i in range(0, len(filtered_tokens), MAX_LENGTH)]
            
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

    def _mecab_tokenize(self, text: str) -> List[str]:
        """Mecab을 사용하여 텍스트를 토큰화"""
        try:
            # 허용할 품사 태그
            VALID_POS = {'NNG', 'NNP'}  # 일반명사, 고유명사만
            
            # 제외할 단어 목록
            STOP_WORDS = {
                # 조사/어미
                '으며', '이며', '하며', '되며', '즐기', '어울리',
                '어야', '어도', '으면', '이면', '아요', '어요',
                '에요', '예요', '으니', '이니', '는데', '은데',
                
                # 일반적인 제외어
                '그것', '이것', '저것', '그런', '이런', '저런',
                '되다', '하다', '있다', '없다', '이다', '아니다',
                '그렇다', '이렇다', '저렇다'
            }

            # 텍스트 전처리
            def preprocess_text(text: str) -> str:
                # 불필요한 패턴 제거
                patterns = [
                    (r'으며\s*', ' '),  # '으며' 제거
                    (r'이며\s*', ' '),  # '이며' 제거
                    (r'하며\s*', ' '),  # '하며' 제거
                    (r'되며\s*', ' '),  # '되며' 제거
                    (r'즐기\s*', ' '),  # '즐기' 제거
                    (r'어울리\s*', ' '), # '어울리' 제거
                    (r'어야\s*', ' '),   # '어야' 제거
                    (r'어도\s*', ' '),   # '어도' 제거
                ]
                
                result = text
                for pattern, repl in patterns:
                    result = re.sub(pattern, repl, result)
                return result
            
            # 텍스트 전처리 수행
            processed_text = preprocess_text(text)
            
            # 형태소 분석
            result = []
            for word, pos in self.mecab.pos(processed_text):
                if (pos in VALID_POS and  # 허용된 품사인지
                    len(word) >= 2 and    # 2글자 이상
                    word not in STOP_WORDS and  # 제외어가 아닌지
                    not word.isdigit()):  # 순수 숫자가 아닌지
                    result.append(word)
            
            # 중복 제거 후 반환
            return list(set(result))
                
        except Exception as e:
            logger.error(f"Mecab 토큰화 중 오류: {str(e)}")
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

    def encode(self, text: str) -> np.ndarray:
        """텍스트를 벡터로 인코딩"""
        try:
            # 입력이 dict나 list인 경우 문자열로 변환
            if isinstance(text, (dict, list)):
                text = json.dumps(text, ensure_ascii=False)
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # CLS 토큰의 마지막 레이어 hidden state를 사용
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                # 1차원 벡터로 변환 (768,)
                embeddings = embeddings.squeeze().astype(np.float32)
                
                # 벡터가 1차원이 아니면 강제로 변환
                if embeddings.ndim != 1:
                    embeddings = embeddings.reshape(-1)
                
                logger.info(f"생성된 벡터 shape: {embeddings.shape}, dtype: {embeddings.dtype}")
                return embeddings
            
        except Exception as e:
            logger.error(f"텍스트 인코딩 중 오류: {str(e)}")
            # 오류 발생 시 768차원의 0 벡터 반환
            return np.zeros(768, dtype=np.float32)