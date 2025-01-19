import os
import json
import logging
from app.models.bert_model import BertEmbedding
from app.utils.article_processor import ArticleProcessor

logger = logging.getLogger(__name__)

class ArticleService:
    def __init__(self):
        self.bert_model = BertEmbedding(os.getenv('MODEL_PATH', './Bert'))
        self.processor = ArticleProcessor(self.bert_model)
        
    def process_folder(self, folder_path: str):
        """폴더 내의 모든 기사를 처리"""
        try:
            if not os.path.exists(folder_path):
                logger.error(f"폴더를 찾을 수 없습니다: {folder_path}")
                return False, "폴더를 찾을 수 없습니다"

            processed_folder = os.path.join(folder_path, 'processed')
            os.makedirs(processed_folder, exist_ok=True)

            processed_files = []
            for filename in os.listdir(folder_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(folder_path, filename)
                    success = self.process_file(file_path)
                    if success:
                        processed_files.append(filename)

            if processed_files:
                return True, f"처리된 파일: {', '.join(processed_files)}"
            return False, "처리할 파일이 없습니다"

        except Exception as e:
            logger.error(f"폴더 처리 중 오류 발생: {str(e)}")
            return False, str(e)

    def process_file(self, file_path: str) -> bool:
        try:
            logger.info(f"=== 파일 처리 시작: {file_path} ===")
            
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)
            
            # 전체 텍스트에서 어려운 단어 추출
            full_text = " ".join(
                block['content'] for block in article['content'] 
                if block['type'] == 'text'
            )
            
            # 전체 텍스트에서 가장 어려운 단어 3개 추출
            difficult_words = self.bert_model.get_difficult_words(full_text)[:3]
            
            # 어려운 단어가 있을 때만 처리
            if difficult_words:
                # 단어 설명을 한 번에 가져오기
                word_descriptions = self.processor.get_batch_descriptions(
                    [{'word': word_info['word']} for word_info in difficult_words]
                )
                
                # 새로운 content 블록 리스트 생성
                new_content = []
                existing_words = set()
                
                # 각 텍스트 블록 처리
                for block in article['content']:
                    if block['type'] == 'text':
                        text = block['content']
                        current_position = 0
                        
                        # 현재 텍스트 블록에서 어려운 단어 찾기
                        for word_info in difficult_words:
                            word = word_info['word']
                            if word not in existing_words:
                                word_position = text.find(word, current_position)
                                if word_position != -1:
                                    # 단어 이전 텍스트 추가
                                    if word_position > current_position:
                                        new_content.append({
                                            "type": "text",
                                            "content": text[current_position:word_position]
                                        })
                                    
                                    # word 블록 추가
                                    new_content.append({
                                        "type": "word",
                                        "content": word,
                                        "description": word_descriptions.get(word, f"{word}에 대한 설명입니다.")
                                    })
                                    
                                    existing_words.add(word)
                                    current_position = word_position + len(word)
                        
                        # 남은 텍스트 추가
                        if current_position < len(text):
                            new_content.append({
                                "type": "text",
                                "content": text[current_position:]
                            })
                    else:
                        new_content.append(block)
                
                article['content'] = new_content
            
            # 어려운 단어가 있든 없든 processed 폴더에 저장
            output_folder = os.path.join(os.path.dirname(file_path), 'processed')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(article, f, ensure_ascii=False, indent=2)
            
            logger.info(f"=== 파일 처리 완료: {file_path} ===")
            return True
            
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {str(e)}")
            return False    