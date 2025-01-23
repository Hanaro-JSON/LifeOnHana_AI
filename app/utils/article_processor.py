import anthropic
import logging
from typing import List, Dict
import os
import json

logger = logging.getLogger(__name__)

class ArticleProcessor:
    def __init__(self, bert_model):
        self.bert_model = bert_model
        self.client = anthropic.Anthropic(
            api_key="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"
        )
        # logger.info("Claude API 클라이언트 초기화 완료")
        # 영구적인 단어 설명 캐시
        self.word_descriptions_cache = {}
        # 캐시 파일 경로
        self.cache_file = os.path.join(os.path.dirname(__file__), 'word_descriptions_cache.json')
        # 캐시 로드
        self._load_cache()

    def _load_cache(self):
        """캐시 파일에서 단어 설명 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.word_descriptions_cache = json.load(f)
                logger.info(f"단어 설명 캐시 로드 완료 (단어 수: {len(self.word_descriptions_cache)})")
        except Exception as e:
            logger.error(f"캐시 로드 중 오류: {str(e)}")

    def _save_cache(self):
        """캐시를 파일에 저장"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.word_descriptions_cache, f, ensure_ascii=False, indent=2)
            logger.info("단어 설명 캐시 저장 완료")
        except Exception as e:
            logger.error(f"캐시 저장 중 오류: {str(e)}")

    def _get_batch_descriptions(self, words: List[Dict[str, str]]) -> Dict[str, str]:
        """단어 설명을 배치로 가져옵니다."""
        # 캐시되지 않은 단어들만 필터링
        uncached_words = [w for w in words if w['word'] not in self.word_descriptions_cache]
        
        if not uncached_words:
            # 모든 단어가 캐시에 있는 경우
            return {w['word']: self.word_descriptions_cache[w['word']] for w in words}
            
        try:
            # 캐시되지 않은 단어들에 대해서만 API 호출
            message = self.client.messages.create(
                model="claude-2.0",
                max_tokens=300,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": f"다음 단어들의 의미를 최대한 간단히 설명해주세요 (각 단어당 10단어 이내):\n{', '.join(w['word'] for w in uncached_words)}"
                }]
            )
            
            if message.content:
                # 새로운 설명들을 캐시에 추가
                descriptions = self._parse_descriptions(message.content[0].text)
                self.word_descriptions_cache.update(descriptions)
                # 캐시 저장
                self._save_cache()
            
            # 전체 요청된 단어들에 대한 설명 반환 (캐시 + 새로운 설명)
            return {w['word']: self.word_descriptions_cache.get(w['word'], f"{w['word']}에 대한 설명") for w in words}
            
        except Exception as e:
            logger.error(f"단어 설명 생성 중 오류: {str(e)}")
            return {w['word']: f"{w['word']}에 대한 설명" for w in words}

    def _parse_descriptions(self, content):
        descriptions = {}
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if ':' in line:
                word, desc = line.split(':', 1)
                word = word.strip()
                desc = desc.strip()
                if word and desc:
                    descriptions[word] = desc
        return descriptions

    def get_batch_descriptions(self, words: List[Dict[str, str]]) -> Dict[str, str]:
        """여러 단어에 대한 설명을 한 번에 가져오기"""
        try:
            descriptions = {}
            words_to_process = []
            
            # 캐시 확인
            for word_dict in words:
                word = word_dict['word']
                if word in self.word_descriptions_cache:
                    descriptions[word] = self.word_descriptions_cache[word]
                else:
                    words_to_process.append(word)
            
            if not words_to_process:
                return descriptions
            
            # API 호출을 위한 프롬프트 생성
            prompt = "다음 단어들에 대해 간단하고 명확한 설명을 해주세요:\n\n"
            for word in words_to_process:
                prompt += f"- {word}\n"
            prompt += "\n각 단어에 대해 한 문장으로 설명해주세요."
            
            # API 호출
            response = self.client.messages.create(
                model="claude-2.0",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # 응답 파싱
            response_text = response.content[0].text
            
            # 응답에서 각 단어별 설명 추출
            for word in words_to_process:
                # 응답에서 해당 단어가 포함된 줄 찾기
                for line in response_text.split('\n'):
                    if word in line:
                        description = line.split(':', 1)[-1].strip()
                        descriptions[word] = description
                        self.word_descriptions_cache[word] = description
                        break
                else:
                    # 설명을 찾지 못한 경우 기본값 설정
                    descriptions[word] = f"{word}에 대한 설명"
            
            # 캐시 저장
            self._save_cache()
            
            return descriptions
            
        except Exception as e:
            logger.error(f"단어 설명 가져오기 중 오류: {str(e)}")
            return {word['word']: f"{word['word']}에 대한 설명" for word in words} 