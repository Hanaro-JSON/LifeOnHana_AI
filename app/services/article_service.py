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

            # 처리된 파일을 저장할 'processed' 폴더 생성
            processed_folder = os.path.join(folder_path, 'processed')
            os.makedirs(processed_folder, exist_ok=True)

            processed_files = []
            for filename in os.listdir(folder_path):
                if filename.endswith('.json'):
                    success = self._process_single_file(
                        os.path.join(folder_path, filename),
                        processed_folder
                    )
                    if success:
                        processed_files.append(filename)

            if processed_files:
                return True, f"처리된 파일: {', '.join(processed_files)}"
            return False, "처리할 파일이 없습니다"

        except Exception as e:
            logger.error(f"폴더 처리 중 오류 발생: {str(e)}")
            return False, str(e)

    def _process_single_file(self, file_path: str, output_folder: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)

            modified = False
            total_words = 0
            processed_words = set()  # 이미 처리된 단어들을 추적
            
            if 'content' in article:
                new_content = []
                for block in article['content']:
                    if isinstance(block, dict) and block.get('type') == 'text' and total_words < 5:
                        content = block.get('content', '')
                        if content and len(content) > 10:
                            difficult_words = self.bert_model.get_difficult_words(content, 0.7)
                            difficult_words = self._filter_most_difficult(difficult_words)
                            
                            # 중복 단어 제거
                            difficult_words = [
                                word for word in difficult_words 
                                if word['word'] not in processed_words
                            ]
                            
                            remaining_words = 5 - total_words
                            difficult_words = difficult_words[:remaining_words]
                            
                            if difficult_words:
                                descriptions = self.processor._get_batch_descriptions(difficult_words)
                                current_pos = 0
                                
                                for word in difficult_words:
                                    if word['word'] in descriptions:
                                        word_pos = content.find(word['word'], current_pos)
                                        if word_pos != -1:
                                            if word_pos > current_pos:
                                                new_content.append({
                                                    "type": "text",
                                                    "content": content[current_pos:word_pos]
                                                })
                                            
                                            new_content.append({
                                                "type": "word",
                                                "content": word['word'],
                                                "description": descriptions[word['word']]
                                            })
                                            
                                            current_pos = word_pos + len(word['word'])
                                            processed_words.add(word['word'])  # 처리된 단어 추가
                                            total_words += 1
                                            modified = True
                                
                                # 남은 텍스트 추가
                                if current_pos < len(content):
                                    new_content.append({
                                        "type": "text",
                                        "content": content[current_pos:]
                                    })
                            else:
                                new_content.append(block)
                        else:
                            new_content.append(block)
                    else:
                        new_content.append(block)
                    
                    if total_words >= 5:  # 최대 단어 수 도달하면 중단
                        break

                # 나머지 블록들 추가
                if total_words >= 5:
                    new_content.extend(article['content'][len(new_content):])
                
                article['content'] = new_content

            if modified:
                output_filename = f"processed_{os.path.basename(file_path)}"
                output_path = os.path.join(output_folder, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)
                
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {str(e)}")
            return False    