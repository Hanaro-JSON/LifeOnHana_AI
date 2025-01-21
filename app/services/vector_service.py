from redis.asyncio import Redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, TypeVar, Optional, Tuple
import logging  # 추가
import math
import json  # 상단에 import 추가
from functools import lru_cache
from config import settings  # 설정 파일 import
import numpy.typing as npt
from flask import g
import pymysql
import torch

# numpy array 타입 정의
ArrayType = TypeVar("ArrayType", bound=npt.NDArray[np.float32])

# 로그 레벨 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorService:
    _instance = None
    _initialized = False
    
    def __new__(cls, redis_client=None, bert_model=None):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, redis_client=None, bert_model=None):
        """벡터 서비스 초기화 (한 번만 실행)"""
        if not self._initialized:
            self.redis = redis_client
            self.bert = bert_model or BertService()
            
            # Redis 연결 확인
            try:
                self.redis.ping()
            except Exception as e:
                logger.error(f"Redis 연결 실패: {str(e)}")
                raise e
                
            # 벡터 인덱스 존재 여부 확인
            try:
                self.redis.execute_command('FT.INFO', 'article_idx')
                logger.info("벡터 인덱스가 이미 존재함")
            except Exception as e:
                if 'Unknown index name' in str(e):
                    logger.info("벡터 인덱스 없음 - 초기화 시작")
                    self.initialize_vectors()
                else:
                    logger.error(f"벡터 인덱스 확인 중 에러: {str(e)}")
                    raise e
            
            self.__class__._initialized = True
            logger.info("VectorService 초기화 완료")
            
            self.async_redis = Redis(
                host=settings.REDIS_CONFIG['host'],
                port=settings.REDIS_CONFIG['port'],
                db=settings.REDIS_CONFIG['db']
            )
            self.VECTOR_DIM = settings.VECTOR_DIM
            self.ab_test_groups = settings.AB_TEST_GROUPS
            self.context_weights = settings.CONTEXT_WEIGHTS
            self.user_actions_key = "user_actions:{}"
            
            # MySQL 연결 설정 추가
            self.db_config = {
                'host': settings.MYSQL_CONFIG['host'],
                'user': settings.MYSQL_CONFIG['user'],
                'password': settings.MYSQL_CONFIG['password'],
                'db': settings.MYSQL_CONFIG['database'],
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor
            }

    def _create_index(self):
        try:
            self.redis.ft("article_idx").create_index([
                VectorField("embedding",
                    "HNSW",  # FLAT 대신 HNSW 사용
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        # HNSW 특정 파라미터
                        "M": 40,  # 각 레이어의 최대 이웃 수
                        "EF_CONSTRUCTION": 200  # 인덱스 구축 시 탐색 범위
                    }
                )
            ], definition=IndexDefinition(prefix=["article:"], index_type=IndexType.HASH))
            
            self.redis.ft("user_idx").create_index([
                TextField("user_id"),
                TextField("article_id"),
                TextField("action_type"),  # view, like, share
                TextField("timestamp")
            ], definition=IndexDefinition(prefix=["user_action:"], index_type=IndexType.HASH))
            
            logger.info("Vector 인덱스 생성 완료 (HNSW)")
        except Exception as e:
            logger.error(f"인덱스 생성 중 에러: {str(e)}")
            logger.info("이미 인덱스가 존재할 수 있음")
    
    def store_article(self, article_id, title, content):
        """기사 벡터 임베딩 저장"""
        try:
            logger.info(f"=== 기사 벡터 저장 시작 === article_id: {article_id}")
            
            # 임베딩 생성
            embedding = self.bert.get_embedding(content)
            logger.info(f"임베딩 생성 완료: shape={embedding.shape}")
            
            # Redis에는 벡터 데이터만 저장
            result = self.redis.hset(
                f"article:{article_id}",
                mapping={
                    "embedding": embedding.tobytes()
                }
            )
            logger.info(f"Redis 벡터 저장 완료: result={result}")
            
            return True
            
        except Exception as e:
            logger.error(f"기사 벡터 저장 중 에러 발생: {str(e)}", exc_info=True)
            raise e
    
    def search_similar(self, query_text, k=5):
        """벡터 유사도 기반 검색"""
        try:
            query_embedding = self.bert.get_embedding(query_text)
            
            # Redis에서 유사한 벡터 검색
            query = (
                Query(
                    f"(*)=>[KNN {k} @embedding $vec_param AS score]"
                )
                .sort_by("score")
                .return_fields("score")  # 벡터 유사도 점수만 반환
                .dialect(2)
                .paging(0, k)
            )
            
            results = self.redis.ft("article_idx").search(
                query,
                {
                    "vec_param": query_embedding.tobytes()
                }
            )

            # MySQL에서 기사 정보 조회
            similar_articles = []
            db = g.db
            with db.cursor() as cursor:
                for doc in results.docs:
                    article_id = doc.id.split(':')[1]
                    cursor.execute("""
                        SELECT article_id, title, content, category 
                        FROM article 
                        WHERE article_id = %s
                    """, (article_id,))
                    article = cursor.fetchone()
                    if article:
                        article['score'] = float(doc.score)
                        similar_articles.append(article)

            return similar_articles
            
        except Exception as e:
            logger.error(f"유사 기사 검색 중 에러: {str(e)}")
            raise e
    
    def record_user_action(self, user_id: str, article_id: str, action_type: str):
        """사용자 행동을 Sorted Set으로 저장"""
        try:
            timestamp = datetime.now().timestamp()
            key = self.user_actions_key.format(user_id)
            
            # Sorted Set에 저장 (score는 timestamp)
            self.redis.zadd(key, {
                f"{article_id}:{action_type}": timestamp
            })
            
            # 30일 이전 데이터 자동 삭제
            old_timestamp = (datetime.now() - timedelta(days=30)).timestamp()
            self.redis.zremrangebyscore(key, '-inf', old_timestamp)
            
        except Exception as e:
            logger.error(f"사용자 행동 기록 중 에러: {str(e)}")
            raise e
    
    def get_user_profile(self, user_id, days=30):
        actions = self._get_recent_actions(user_id, days)
        
        # 행동별 + 시간 가중치
        action_weights = {
            "view": 1.0,
            "like": 2.0,
            "share": 3.0
        }
        
        profile_vector = np.zeros(self.VECTOR_DIM)
        now = datetime.now()
        
        for action in actions:
            # 시간 가중치 계산 (최근 행동일수록 높은 가중치)
            action_time = datetime.fromisoformat(action['timestamp'])
            days_old = (now - action_time).days
            time_weight = 1.0 / (1.0 + days_old * 0.1)  # 시간 감쇠
            
            # 최종 가중치 = 행동 가중치 * 시간 가중치
            article_vector = self._get_article_vector(action['article_id'])
            weight = action_weights.get(action['action_type'], 1.0) * time_weight
            profile_vector += article_vector * weight
        
        if np.any(profile_vector):
            profile_vector = profile_vector / np.linalg.norm(profile_vector)
        
        return profile_vector 
    
    def get_diverse_recommendations(self, user_id, k=10, diversity_weight=0.3):
        # 기본 추천 받기
        base_results = self.search_similar_by_vector(
            self.get_user_profile(user_id), 
            k=k*2  # 더 많은 후보 추출
        )
        
        # 카테고리 다양성 확보
        selected_articles = []
        categories_used = set()
        
        for doc in base_results.docs:
            category = self._get_article_category(doc.title, doc.content)
            
            # 새로운 카테고리면 가중치 부여
            diversity_score = 1.0
            if category not in categories_used:
                diversity_score += diversity_weight
                categories_used.add(category)
            
            # 최종 점수 = 유사도 * 다양성 점수
            final_score = (1 - float(doc.score)) * diversity_score
            
            selected_articles.append({
                "title": doc.title,
                "content": doc.content,
                "similarity": final_score,
                "category": category
            })
        
        # 최종 점수로 정렬하고 상위 k개 선택
        return sorted(selected_articles, 
                     key=lambda x: x['similarity'], 
                     reverse=True)[:k] 
    
    def get_hybrid_recommendations(self, user_id, k=10):
        # 1. 컨텐츠 기반 추천 (현재 방식)
        content_recs = self.get_diverse_recommendations(user_id, k=k)
        
        # 2. 협업 필터링 추천
        similar_users = self._find_similar_users(user_id)
        cf_recs = self._get_cf_recommendations(similar_users)
        
        # 3. 두 추천 결과 통합
        final_recs = []
        content_weight = 0.7  # 컨텐츠 기반 가중치
        cf_weight = 0.3      # 협업 필터링 가중치
        
        seen_articles = set()
        
        for content_rec in content_recs:
            article_id = content_rec['article_id']
            if article_id in seen_articles:
                continue
            
            # CF 점수 찾기
            cf_score = next(
                (r['score'] for r in cf_recs if r['article_id'] == article_id), 
                0.0
            )
            
            # 최종 점수 계산
            final_score = (
                content_weight * content_rec['similarity'] +
                cf_weight * cf_score
            )
            
            final_recs.append({
                **content_rec,
                'final_score': final_score
            })
            seen_articles.add(article_id)
        
        return sorted(final_recs, key=lambda x: x['final_score'], reverse=True) 
    
    def get_recommendations(self, user_id: str, seed: str = None, k: int = 50) -> List[str]:
        """사용자 맞춤 기사 추천"""
        try:
            logger.info(f"=== 추천 시작 === user_id: {user_id}, k: {k}")
            
            # AB 테스트 그룹 결정
            ab_group = self._get_ab_test_group(user_id)
            logger.info(f"AB 테스트 그룹: {ab_group}")
            
            # 최근 본 기사들 조회
            recent_actions = self._get_recent_actions(user_id, hours=24)
            logger.info(f"최근 행동 수: {len(recent_actions)}")
            
            # 컨텐츠 기반 추천
            content_scores = {}
            
            # 최근 행동이 없으면 전체 기사에서 최신순으로 추천
            if not recent_actions:
                logger.info("최근 행동 없음 - 전체 기사에서 추천")
                try:
                    with self._get_db_connection() as db:
                        with db.cursor() as cursor:
                            cursor.execute("""
                                SELECT article_id 
                                FROM article 
                                ORDER BY published_at DESC 
                                LIMIT %s
                            """, (k * 2,))
                            results = cursor.fetchall()
                            for result in results:
                                content_scores[str(result['article_id'])] = 1.0
                except Exception as e:
                    logger.error(f"최신 기사 조회 중 에러: {str(e)}")
            else:
                logger.info("최근 행동 기반 추천 시작")
                for action in recent_actions:
                    article_id = str(action['article_id'])
                    logger.info(f"기사 {article_id}의 유사 기사 검색")
                    
                    # 유사한 기사 검색
                    vector = self._get_article_vector(article_id)
                    if vector is None:
                        logger.warning(f"기사 {article_id}의 벡터를 찾을 수 없음")
                        continue
                        
                    similar_articles = self.search_similar_by_vector(vector, k)
                    logger.info(f"기사 {article_id}의 유사 기사: {similar_articles}")
                    
                    # 점수 합산
                    for similar_id, score in similar_articles:
                        if similar_id not in content_scores:
                            content_scores[similar_id] = 0
                        content_scores[similar_id] += score
            
            # 점수로 정렬하여 상위 k개 반환
            recommended = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            logger.info(f"최종 추천 결과: {recommended}")
            return [article_id for article_id, _ in recommended]
            
        except Exception as e:
            logger.error(f"추천 중 에러 발생: {str(e)}")
            return []

    def _get_session_interests(self, user_id: str) -> Dict[str, float]:
        """최근 세션의 관심사 분석"""
        recent_actions = self._get_recent_actions(user_id, hours=1)
        interests = {}
        
        for action in recent_actions:
            category = self._get_article_category(action['article_id'])
            interests[category] = interests.get(category, 0) + 1
        
        # 정규화
        if interests:
            max_count = max(interests.values())
            interests = {k: v/max_count for k, v in interests.items()}
        
        return interests

    def _calculate_base_score(self, content_rec: Dict, cf_recs: List[Dict], 
                            weights: Dict, session_interests: Dict) -> float:
        """기본 점수 계산 로직"""
        # 1. 컨텐츠 기반 점수
        content_score = content_rec['similarity'] * weights['content']
        
        # 2. 협업 필터링 점수
        cf_score = next(
            (r['score'] for r in cf_recs if r['article_id'] == content_rec['article_id']),
            0.0
        ) * weights['cf']
        
        # 3. 시간 가중치
        time_score = self._calculate_time_weight(content_rec['timestamp']) * weights['time']
        
        # 4. 다양성 점수
        diversity_score = (1.0 if content_rec['category'] not in self.seen_categories 
                         else 0.0) * weights['diversity']
        
        # 5. 세션 관심사 반영
        session_score = session_interests.get(content_rec['category'], 0.0) * 0.2
        
        return content_score + cf_score + time_score + diversity_score + session_score

    async def _log_recommendation_results(self, user_id: str, recommendations: List[Dict], ab_group: str):
        """비동기 로깅"""
        try:
            timestamp = datetime.now().isoformat()
            log_data = {
                'user_id': user_id,
                'timestamp': timestamp,
                'ab_group': ab_group,
                'recommendations': json.dumps([
                    {
                        'article_id': rec['article_id'],
                        'score': float(rec['final_score']),
                        'category': rec['category']
                    } for rec in recommendations
                ])
            }
            
            # 비동기로 로그 저장
            await self.async_redis.hset(
                f"recommendation_log:{user_id}:{timestamp}",
                mapping=log_data
            )
            
        except Exception as e:
            logger.error(f"로깅 중 에러: {str(e)}")

    def _get_ab_test_group(self, user_id: str) -> str:
        """사용자의 AB 테스트 그룹 결정"""
        try:
            # 기존 그룹 조회
            group = self.redis.get(f"ab_test:{user_id}")
            
            # bytes를 문자열로 디코딩
            if isinstance(group, bytes):
                group = group.decode('utf-8')
            
            # 그룹이 없으면 새로 할당
            if not group:
                group = 'A' if hash(user_id) % 2 == 0 else 'B'
                self.redis.set(f"ab_test:{user_id}", group)
            
            return group
            
        except Exception as e:
            logger.error(f"AB 테스트 그룹 결정 중 에러: {str(e)}")
            return 'A'  # 에러 시 기본 그룹 반환

    def _get_time_of_day(self) -> str:
        """시간대 확인"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    def _find_similar_users(self, user_id: str) -> List[str]:
        """유사한 사용자 찾기"""
        user_vector = self.get_user_profile(user_id)
        similar_users = []
        
        # Redis에서 모든 사용자 프로필 검색
        for key in self.redis.scan_iter("user_profile:*"):
            other_id = key.decode().split(':')[1]
            if other_id != user_id:
                other_vector = np.frombuffer(self.redis.get(key), dtype=np.float32)
                similarity = np.dot(user_vector, other_vector)
                similar_users.append((other_id, similarity))
        
        # 상위 10명의 유사 사용자 반환
        return [uid for uid, _ in sorted(similar_users, key=lambda x: x[1], reverse=True)[:10]]
    
    def _get_cf_recommendations(self, similar_users: List[str]) -> List[Dict]:
        """협업 필터링 기반 추천"""
        cf_scores = {}
        
        # 유사 사용자들의 행동 기록 분석
        for user_id in similar_users:
            actions = self._get_recent_actions(user_id)
            for action in actions:
                article_id = action['article_id']
                weight = {
                    'view': 1.0,
                    'like': 2.0,
                    'share': 3.0
                }.get(action['action_type'], 1.0)
                
                if article_id not in cf_scores:
                    cf_scores[article_id] = 0.0
                cf_scores[article_id] += weight
        
        # 정규화
        if cf_scores:
            max_score = max(cf_scores.values())
            cf_scores = {k: v/max_score for k, v in cf_scores.items()}
        
        return [{'article_id': k, 'score': v} for k, v in cf_scores.items()] 

    def _get_recent_actions(self, user_id: str, hours: int = 24) -> List[Dict]:
        """Redis와 MySQL에서 사용자의 좋아요한 기사들 조회"""
        try:
            # 1. 먼저 Redis 확인
            user_likes_key = f"user:{user_id}:likes"
            liked_articles_map = self.redis.hgetall(user_likes_key)
            
            liked_article_ids = [
                article_id for article_id, is_liked in liked_articles_map.items()
                if is_liked == b'true' or is_liked == b'True' or is_liked == b'1'
            ]
            
            # 2. Redis에 없으면 MySQL에서 조회
            if not liked_article_ids:
                logger.info("Redis에 좋아요 정보가 없어 MySQL 확인")
                with self._get_db_connection() as db:
                    with db.cursor() as cursor:
                        cursor.execute("""
                            SELECT a.article_id, a.category, a.published_at
                            FROM article a
                            JOIN article_like al ON a.article_id = al.article_id
                            WHERE al.user_id = %s 
                            AND al.is_like = TRUE
                            ORDER BY a.published_at DESC
                        """, (user_id,))
                        liked_articles = cursor.fetchall()
                        
                        # MySQL 결과를 Redis에도 캐시
                        if liked_articles:
                            for article in liked_articles:
                                self.redis.hset(user_likes_key, str(article['article_id']), 'true')
                            # 캐시 만료시간 설정 (예: 1시간)
                            self.redis.expire(user_likes_key, 3600)
                        
                        return liked_articles
            
            # Redis에 있으면 해당 article_ids로 MySQL 조회
            else:
                with self._get_db_connection() as db:
                    with db.cursor() as cursor:
                        placeholders = ', '.join(['%s'] * len(liked_article_ids))
                        cursor.execute(f"""
                            SELECT article_id, category, published_at
                            FROM article 
                            WHERE article_id IN ({placeholders})
                            ORDER BY published_at DESC
                        """, liked_article_ids)
                        return cursor.fetchall()
                
        except Exception as e:
            logger.error(f"최근 행동 조회 중 에러: {str(e)}")
            return []

    def _get_article_like_count(self, article_id: str) -> int:
        """기사의 좋아요 수 조회"""
        try:
            article_like_count_key = f"article:{article_id}:likeCount"
            # Redis에서 좋아요 수 조회
            like_count = self.redis.get(article_like_count_key)
            
            if like_count is None:
                # Redis에 없으면 MySQL에서 조회하여 캐시
                with self._get_db_connection() as db:
                    with db.cursor() as cursor:
                        cursor.execute("""
                            SELECT like_count 
                            FROM article 
                            WHERE article_id = %s
                        """, (article_id,))
                        result = cursor.fetchone()
                        like_count = result['like_count'] if result else 0
                        
                # Redis에 캐시
                self.redis.set(article_like_count_key, like_count)
            
            return int(like_count)
            
        except Exception as e:
            logger.error(f"좋아요 수 조회 중 에러: {str(e)}")
            return 0

    def _get_article_category(self, article_id: str) -> str:
        """MySQL에서 기사 카테고리 조회"""
        try:
            with self._get_db_connection() as db:
                with db.cursor() as cursor:
                    cursor.execute("""
                        SELECT category 
                        FROM article 
                        WHERE article_id = %s
                    """, (article_id,))
                    result = cursor.fetchone()
                    
                    if result and result.get('category'):
                        return result['category']
                        
                    logger.warning(f"기사 {article_id}의 카테고리를 찾을 수 없습니다.")
                    return 'UNKNOWN'
        except Exception as e:
            logger.error(f"카테고리 조회 중 에러: {str(e)}")
            return 'UNKNOWN'

    def get_similar_articles(self, article_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """특정 기사와 유사한 기사 검색"""
        try:
            # 기사 벡터 가져오기
            vector = self._get_article_vector(article_id)
            if vector is None:
                logger.error(f"기사 {article_id}의 벡터를 찾을 수 없습니다.")
                return []
            
            # 유사한 기사 검색
            similar_articles = self.search_similar_by_vector(vector, k)
            logger.info(f"유사 기사 검색 결과: {len(similar_articles)}개")
            
            # 자기 자신 제외
            similar_articles = [(aid, score) for aid, score in similar_articles if aid != article_id]
            
            return similar_articles
            
        except Exception as e:
            logger.error(f"기사 {article_id} 처리 중 에러: {str(e)}")
            return []

    def search_similar_by_vector(self, vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """벡터로 유사한 기사 검색"""
        try:
            logger.info(f"벡터 검색 시작 - 벡터 shape: {vector.shape}")
            logger.info(f"검색 벡터 타입: {type(vector)}, dtype: {vector.dtype}")
            
            # Redis 벡터 검색 쿼리 실행
            query_vector = vector.tobytes()
            results = self.redis.execute_command(
                'FT.SEARCH', 'article_idx', 
                f'*=>[KNN {k} @embedding $vec AS distance]',
                'PARAMS', '2', 'vec', query_vector,
                'RETURN', '2', 'article_id', 'distance',
                'DIALECT', '2'
            )
            
            logger.info(f"Redis 검색 결과: {results}")
            
            # 결과 파싱
            if results and len(results) > 1:
                similar_articles = []
                for i in range(1, len(results), 2):
                    doc_key = results[i]  # b'article:article2'
                    if isinstance(doc_key, bytes):
                        article_id = doc_key.decode('utf-8').split(':')[1].replace('article', '')
                        # 다음 항목은 [b'distance', b'0.955285251141'] 형식
                        distance = float(results[i+1][1])
                        score = 1.0 - distance  # 거리를 유사도로 변환
                        similar_articles.append((article_id, score))
            
            logger.info(f"파싱된 검색 결과: {similar_articles}")
            return similar_articles
            
        except Exception as e:
            logger.error(f"벡터 검색 중 에러 발생: {str(e)}")
            logger.error(f"에러 발생 시 결과: {results if 'results' in locals() else 'No results'}")
            return []

    @lru_cache(maxsize=1000)
    def _get_article_vector(self, article_id: str) -> Optional[np.ndarray]:
        """기사의 벡터 표현 조회 또는 생성"""
        try:
            # Redis 키 생성 시 실제 article_id 사용
            vector_key = f"article:{article_id}"  # "article:45" 형식으로 저장
            vector_data = self.redis.get(vector_key)
            
            if vector_data:
                return np.frombuffer(vector_data, dtype=np.float32)
            
            # Redis에 없으면 MySQL에서 기사 내용 조회하여 벡터 생성
            with self._get_db_connection() as db:
                with db.cursor() as cursor:
                    cursor.execute("""
                        SELECT title, content 
                        FROM article 
                        WHERE article_id = %s
                    """, (article_id,))
                    article = cursor.fetchone()
                    
                    if not article:
                        raise ValueError(f"Article {article_id} not found in database")
                    
                    # 제목과 내용 추출
                    title = article['title']
                    content = article['content']
                    
                    # JSON 파싱
                    if isinstance(content, str):
                        content = json.loads(content)
                    
                    # text 타입의 내용만 추출하여 합치기
                    text_contents = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_contents.append(item.get('content', ''))
                            # word 타입도 포함 (설명과 함께)
                            elif item.get('type') == 'word':
                                word = item.get('content', '')
                                desc = item.get('description', '')
                                if word and desc:
                                    text_contents.append(f"{word} - {desc}")
                    
                    # 최종 텍스트 생성 (제목 + 본문)
                    full_text = f"{title} {' '.join(text_contents)}"
                    
                    # ALBERT 토크나이저로 토큰화
                    tokens = self.bert.tokenizer.tokenize(full_text)
                    if not tokens:
                        raise ValueError("토큰화 결과가 없습니다")
                        
                    # 토큰을 ID로 변환
                    token_ids = self.bert.tokenizer.convert_tokens_to_ids(tokens)
                    
                    # 최대 길이 제한 (config의 max_position_embeddings 참고)
                    if len(token_ids) > 512:
                        token_ids = token_ids[:512]
                    
                    # 모델 입력을 위한 텐서 생성
                    input_ids = torch.tensor([token_ids])
                    
                    # ALBERT 모델로 벡터 생성
                    with torch.no_grad():
                        outputs = self.bert.model(input_ids)
                        # [CLS] 토큰의 마지막 히든 스테이트를 벡터로 사용
                        vector = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)[0]
                    
                    # Redis에 벡터 캐시 (실제 article_id 사용)
                    self.redis.set(vector_key, vector.tobytes())
                    self.redis.expire(vector_key, 86400)  # 24시간 캐시
                    
                    return vector
                
        except Exception as e:
            logger.error(f"기사 벡터 조회 중 에러: {str(e)}")
            return None

    def _calculate_time_weight(self, timestamp: str) -> float:
        """시간 기반 가중치 계산"""
        try:
            if not timestamp:
                return 0.5
            
            # timestamp가 bytes 타입인 경우 디코딩
            if isinstance(timestamp, bytes):
                timestamp = timestamp.decode('utf-8')
            
            time_diff = datetime.now() - datetime.fromisoformat(str(timestamp))
            hours_old = time_diff.total_seconds() / 3600
            
            # 24시간 이내: 1.0 ~ 0.5
            # 24시간 이후: 0.5 ~ 0.1
            if hours_old <= 24:
                return 1.0 - (hours_old / 48)  # 24시간동안 선형 감소
            else:
                return max(0.1, 0.5 * math.exp(-0.1 * (hours_old - 24)))
            
        except Exception as e:
            logger.error(f"시간 가중치 계산 중 에러: {str(e)}")
            return 0.5  # 에러 발생 시 기본값 반환

    def _check_and_create_index(self):
        """인덱스 확인 및 생성"""
        try:
            # 기존 인덱스 정보 확인
            try:
                info = self.redis.ft("article_idx").info()
                logger.info(f"기존 인덱스 정보: {info}")
            except:
                logger.info("기존 인덱스 없음")
                info = None
            
            # 인덱스가 없으면 생성
            if not info:
                logger.info("새 인덱스 생성 시작")
                try:
                    self.redis.ft("article_idx").create_index([
                        TextField("title"),
                        TextField("content"),
                        VectorField("embedding",
                            "HNSW",  # FLAT 대신 HNSW 사용
                            {
                                "TYPE": "FLOAT32",
                                "DIM": self.VECTOR_DIM,
                                "DISTANCE_METRIC": "COSINE",
                                "INITIAL_CAP": 1000,
                                # HNSW 특정 파라미터
                                "M": 40,  # 각 레이어의 최대 이웃 수
                                "EF_CONSTRUCTION": 200  # 인덱스 구축 시 탐색 범위
                            }
                        )
                    ], definition=IndexDefinition(prefix=["article:"], index_type=IndexType.HASH))
                    logger.info("새 인덱스 생성 완료")
                except Exception as e:
                    logger.error(f"인덱스 생성 중 에러: {str(e)}")
                    raise e
                
            # 인덱스에 포함된 문서 수 확인
            try:
                doc_count = self.redis.ft("article_idx").info()['num_docs']
                logger.info(f"인덱스 문서 수: {doc_count}")
            except:
                logger.error("문서 수 확인 실패")
            
        except Exception as e:
            logger.error(f"인덱스 확인 중 에러: {str(e)}")
            raise e 

    def _get_db_connection(self):
        """MySQL 연결 생성"""
        return pymysql.connect(**self.db_config)

    def _get_articles_from_db(self, article_ids: List[str]) -> List[Dict]:
        """MySQL에서 기사 정보 가져오기"""
        try:
            if not article_ids:
                return []
            
            with self._get_db_connection() as db:
                with db.cursor() as cursor:
                    placeholders = ', '.join(['%s'] * len(article_ids))
                    query = f"""
                        SELECT article_id, title, content, category, published_at, like_count 
                        FROM article 
                        WHERE article_id IN ({placeholders})
                        ORDER BY FIELD(article_id, {placeholders})
                    """
                    cursor.execute(query, article_ids + article_ids)
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"MySQL에서 기사 조회 중 에러: {str(e)}")
            return [] 

    def _get_article_metadata(self, article_id: str) -> Optional[Dict]:
        """기사의 메타데이터(카테고리, 발행일 등) 조회"""
        try:
            # Redis 문서 ID에서 숫자만 추출 (예: "article:45" -> "45")
            if ':' in article_id:
                article_id = article_id.split(':')[1]
            
            with self._get_db_connection() as db:
                with db.cursor() as cursor:
                    cursor.execute("""
                        SELECT article_id, category, published_at
                        FROM article 
                        WHERE article_id = %s
                    """, (article_id,))
                    article = cursor.fetchone()
                    
                    if article:
                        return {
                            'article_id': article['article_id'],
                            'category': article['category'],
                            'published_at': article['published_at']
                        }
                    return None
                
        except Exception as e:
            logger.error(f"기사 메타데이터 조회 중 에러: {str(e)}")
            return None 

    def _check_redis_data(self):
        """Redis에 저장된 데이터 확인"""
        try:
            # 모든 키 조회
            keys = self.redis.keys('article:*')
            logger.info(f"Redis에 저장된 총 기사 수: {len(keys)}")
            
            # 인덱스 정보 조회
            try:
                info = self.redis.execute_command('FT.INFO', 'article_idx')
                logger.info(f"Redis 검색 인덱스 정보: {info}")
            except Exception as e:
                logger.error(f"검색 인덱스 조회 실패: {str(e)}")
            
            # 몇 개의 키 샘플 출력
            for key in keys[:3]:
                try:
                    data = self.redis.hgetall(key)
                    logger.info(f"키 {key}의 데이터: {data}")
                except Exception as e:
                    logger.error(f"키 {key} 데이터 조회 실패: {str(e)}")
                
        except Exception as e:
            logger.error(f"Redis 데이터 확인 중 에러: {str(e)}")

    def initialize_vectors(self):
        """모든 기사의 벡터를 생성하여 Redis에 저장"""
        try:
            logger.info("=== 기사 벡터 초기화 시작 ===")
            
            # Redis 검색 인덱스 생성
            try:
                self.redis.execute_command(
                    'FT.CREATE', 'article_idx', 'ON', 'HASH',
                    'PREFIX', '1', 'article:',
                    'SCHEMA', 'embedding', 'VECTOR', 'FLAT', '6', 
                    'TYPE', 'FLOAT32', 'DIM', '768', 
                    'DISTANCE_METRIC', 'COSINE',
                    'article_id', 'TEXT'
                )
            except Exception as e:
                if 'Index already exists' not in str(e):
                    raise e
            
            # Redis에 저장된 기사 ID 목록 가져오기
            existing_articles = set()
            for key in self.redis.keys('article:*'):
                try:
                    article_id = int(key.decode('utf-8').split(':')[1])
                    existing_articles.add(article_id)
                except:
                    continue
            
            # MySQL에서 아직 벡터가 생성되지 않은 기사만 조회
            with self._get_db_connection() as db:
                with db.cursor() as cursor:
                    cursor.execute("""
                        SELECT article_id, title, content 
                        FROM article
                        ORDER BY article_id
                    """)
                    articles = cursor.fetchall()
                    
                    # 이미 벡터가 있는 기사 필터링
                    articles = [
                        article for article in articles 
                        if article['article_id'] not in existing_articles
                    ]
                    
                    if not articles:
                        logger.info("모든 기사의 벡터가 이미 생성되어 있습니다.")
                        return
                    
                    total = len(articles)
                    logger.info(f"총 {total}개 기사 처리 시작")
                    
                    for i, article in enumerate(articles, 1):
                        try:
                            article_id = article['article_id']
                            logger.info(f"기사 {article_id} 처리 중... ({i}/{total})")
                            
                            # Redis에 이미 있는지 확인
                            vector_key = f"article:{article_id}"
                            if self.redis.exists(vector_key):
                                logger.info(f"기사 {article_id}는 이미 처리됨, 건너뜀")
                                continue
                            
                            title = article['title']
                            content = article['content']
                            
                            # JSON 파싱
                            if isinstance(content, str):
                                content = json.loads(content)
                            
                            # text 타입의 내용만 추출하여 합치기
                            text_contents = []
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get('type') == 'text':
                                        text_contents.append(item.get('content', ''))
                                    elif item.get('type') == 'word':
                                        word = item.get('content', '')
                                        desc = item.get('description', '')
                                        if word and desc:
                                            text_contents.append(f"{word} - {desc}")
                            
                            # 최종 텍스트 생성 (제목 + 본문)
                            full_text = f"{title} {' '.join(text_contents)}"
                            logger.info(f"기사 {article_id} 텍스트 추출 완료 (길이: {len(full_text)})")
                            
                            # ALBERT 토크나이저로 토큰화
                            tokens = self.bert.tokenizer.tokenize(full_text)
                            if not tokens:
                                raise ValueError("토큰화 결과가 없습니다")
                            logger.info(f"기사 {article_id} 토큰화 완료 (토큰 수: {len(tokens)})")
                                
                            # 토큰을 ID로 변환
                            token_ids = self.bert.tokenizer.convert_tokens_to_ids(tokens)
                            
                            # 최대 길이 제한
                            if len(token_ids) > 512:
                                token_ids = token_ids[:512]
                                logger.info(f"기사 {article_id} 토큰 길이 제한 (512)")
                            
                            # 모델 입력을 위한 텐서 생성
                            input_ids = torch.tensor([token_ids])
                            
                            # ALBERT 모델로 벡터 생성
                            logger.info(f"기사 {article_id} 벡터 생성 중...")
                            with torch.no_grad():
                                outputs = self.bert.model(input_ids)
                                vector = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)[0]
                            
                            # Redis에 HASH 형식으로 벡터 저장
                            self.redis.hset(
                                vector_key,
                                mapping={
                                    'embedding': vector.tobytes(),
                                    'article_id': str(article_id)
                                }
                            )
                            logger.info(f"기사 {article_id} 벡터 저장 완료")
                            
                        except Exception as e:
                            logger.error(f"기사 {article_id} 처리 중 에러: {str(e)}")
                            continue
                    
                    # 초기화 완료 후 데이터 확인
                    self._check_redis_data()
                    logger.info("=== 기사 벡터 초기화 완료 ===")
                    
        except Exception as e:
            logger.error(f"벡터 초기화 중 에러 발생: {str(e)}")
            raise e 