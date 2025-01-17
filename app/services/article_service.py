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
            processed_words = set()
            
            if 'content' in article:
                new_content = []
                for block in article['content']:
                    if isinstance(block, dict) and block.get('type') == 'text' and total_words < 5:
                        content = block.get('content', '')
                        if content and len(content) > 10:
                            # 새로운 난이도 기준 적용 (BERT + Korpora)
                            difficult_words = self.bert_model.get_difficult_words(
                                content, 
                                threshold=0.65  # 빈도 정보가 추가되어 threshold 조정
                            )
                            
                            # 난이도 점수로 정렬하고 중복 제거
                            difficult_words = sorted(
                                [word for word in difficult_words if word['word'] not in processed_words],
                                key=lambda x: x['difficulty_score'],
                                reverse=True
                            )
                            
                            remaining_words = 5 - total_words
                            difficult_words = difficult_words[:remaining_words]
                            
                            if difficult_words:
                                # 디버깅을 위한 난이도 정보 로깅
                                for word in difficult_words:
                                    logger.debug(
                                        f"선택된 단어: {word['word']}, "
                                        f"난이도 점수: {word['difficulty_score']:.3f}, "
                                        f"빈도: {word['frequency']:.6f}, "
                                        f"BERT 점수: {word['bert_score']:.3f}"
                                    )
                                
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
                                                "description": descriptions[word['word']],
                                                "difficulty_info": {  # 난이도 정보 추가
                                                    "score": round(word['difficulty_score'], 3),
                                                    "frequency": round(word['frequency'], 6),
                                                    "bert_score": round(word['bert_score'], 3)
                                                }
                                            })
                                            
                                            current_pos = word_pos + len(word['word'])
                                            processed_words.add(word['word'])
                                            total_words += 1
                                            modified = True
                                
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