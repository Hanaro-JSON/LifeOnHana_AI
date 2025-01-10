from redis.asyncio import Redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, TypeVar
import logging  # 추가
import math
import json  # 상단에 import 추가
from functools import lru_cache
from config import settings  # 설정 파일 import
import numpy.typing as npt

# numpy array 타입 정의
ArrayType = TypeVar("ArrayType", bound=npt.NDArray[np.float32])

# 로그 레벨 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self, redis_client, bert_model):
        self.redis = redis_client
        self.async_redis = Redis(
            host=settings.REDIS_CONFIG['host'],
            port=settings.REDIS_CONFIG['port'],
            db=settings.REDIS_CONFIG['db']
        )
        self.bert = bert_model
        # 설정 파일로 이동
        self.VECTOR_DIM = settings.VECTOR_DIM
        self.ab_test_groups = settings.AB_TEST_GROUPS
        self.context_weights = settings.CONTEXT_WEIGHTS
        
        # Redis Sorted Set 키
        self.user_actions_key = "user_actions:{}"
        
        logger.info("VectorService 초기화 완료")

    def _create_index(self):
        try:
            self.redis.ft("article_idx").create_index([
                TextField("title"),
                TextField("content"),
                VectorField("embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        "BLOCK_SIZE": 100
                    }
                )
            ], definition=IndexDefinition(prefix=["article:"], index_type=IndexType.HASH))

            self.redis.ft("user_idx").create_index([
                TextField("user_id"),
                TextField("article_id"),
                TextField("action_type"),  # view, like, share
                TextField("timestamp")
            ], definition=IndexDefinition(prefix=["user_action:"], index_type=IndexType.HASH))
        except:
            print("Index already exists")
    
    def store_article(self, article_id, title, content):
        """기사 저장"""
        try:
            logger.info(f"=== 기사 저장 시작 === article_id: {article_id}")
            
            # 임베딩 생성
            embedding = self.bert.get_embedding(content)
            logger.info(f"임베딩 생성 완료: shape={embedding.shape}")
            
            # Redis에 저장
            result = self.redis.hset(
                f"article:{article_id}",
                mapping={
                    "title": title,
                    "content": content,
                    "embedding": embedding.tobytes()
                }
            )
            logger.info(f"Redis 저장 완료: result={result}")
            
            return True
            
        except Exception as e:
            logger.error(f"기사 저장 중 에러 발생: {str(e)}", exc_info=True)
            raise e
    
    def search_similar(self, query_text, k=5):
        query_embedding = self.bert.get_embedding(query_text)
        
        query = (
            Query(
                f"(*)=>[KNN {k} @embedding $vec_param AS score]"
            )
            .sort_by("score")
            .return_fields("title", "content", "score")
            .dialect(2)
            .paging(0, k)
        )
        
        results = self.redis.ft("article_idx").search(
            query,
            {
                "vec_param": query_embedding.tobytes()
            }
        )
        return results 
    
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
    
    def get_recommendations(self, user_id: str, context: dict = None, k: int = 10) -> List[Dict]:
        """사용자 맞춤 기사 추천"""
        try:
            logger.info(f"=== 추천 시작 === user_id: {user_id}, k: {k}")
            
            # AB 테스트 그룹 결정
            ab_group = self._get_ab_test_group(user_id)
            logger.info(f"AB 테스트 그룹: {ab_group}")
            
            # 1. 최근 본 기사들 조회
            recent_actions = self._get_recent_actions(user_id, hours=24)
            logger.info(f"최근 행동 수: {len(recent_actions)}")
            logger.info(f"최근 행동: {recent_actions}")
            
            # 2. 컨텐츠 기반 추천
            content_scores = {}
            
            # 최근 행동이 없으면 전체 기사에서 최신순으로 추천
            if not recent_actions:
                logger.info("최근 행동 없음 - 전체 기사에서 추천")
                for key in self.redis.scan_iter("article:*"):
                    try:
                        article_id = key.decode('utf-8').split(':')[1]
                        article_data = self.redis.hgetall(key)
                        
                        if article_data:
                            content_scores[article_id] = 1.0  # 기본 점수
                    except Exception as e:
                        logger.error(f"기사 처리 중 에러: {str(e)}")
                        continue
            else:
                # 최근 본 기사 기반 추천
                logger.info("최근 행동 기반 추천 시작")
                for action in recent_actions:
                    article_id = action.get('article_id')
                    if article_id:
                        try:
                            article_vector = self._get_article_vector(article_id)
                            results = self.search_similar_by_vector(article_vector, k=k)
                            logger.info(f"유사 기사 검색 결과: {len(results.docs)}개")
                            
                            for doc in results.docs:
                                doc_id = doc.id.split(':')[1]
                                score = 1 - float(doc.score)
                                content_scores[doc_id] = max(content_scores.get(doc_id, 0), score)
                        except Exception as e:
                            logger.error(f"기사 {article_id} 처리 중 에러: {str(e)}")
                            continue
            
            # 3. 최종 추천 결과 생성
            recommendations = []
            for article_id, score in content_scores.items():
                try:
                    article_data = self.redis.hgetall(f"article:{article_id}")
                    if article_data:
                        title = article_data.get(b'title', b'').decode('utf-8')
                        content = article_data.get(b'content', b'').decode('utf-8')
                        
                        recommendations.append({
                            'article_id': article_id,
                            'title': title,
                            'content': content,
                            'final_score': score
                        })
                except Exception as e:
                    logger.error(f"기사 데이터 처리 중 에러: {str(e)}")
                    continue
            
            # 점수로 정렬하고 상위 k개 반환
            recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            logger.info(f"최종 추천 결과: {len(recommendations)}개")
            
            logger.info(f"=== 추천 완료 === 결과 수: {len(recommendations[:k])}")
            return recommendations[:k]
            
        except Exception as e:
            logger.error(f"추천 생성 중 에러 발생: {str(e)}", exc_info=True)
            raise e 

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

    def _get_recent_actions(self, user_id: str, days: int = 30, hours: int = None) -> List[Dict]:
        """Sorted Set을 사용한 최근 행동 조회"""
        try:
            key = self.user_actions_key.format(user_id)
            
            # 시간 범위 설정
            if hours:
                min_timestamp = (datetime.now() - timedelta(hours=hours)).timestamp()
            else:
                min_timestamp = (datetime.now() - timedelta(days=days)).timestamp()
                
            # Sorted Set에서 시간 범위로 조회
            action_data = self.redis.zrangebyscore(
                key, 
                min_timestamp, 
                '+inf', 
                withscores=True
            )
            
            actions = []
            for action_str, timestamp in action_data:
                article_id, action_type = action_str.decode().split(':')
                actions.append({
                    'user_id': user_id,
                    'article_id': article_id,
                    'action_type': action_type,
                    'timestamp': datetime.fromtimestamp(timestamp).isoformat()
                })
            
            return actions
            
        except Exception as e:
            logger.error(f"최근 행동 조회 중 에러: {str(e)}")
            return []

    def _get_article_category(self, title: str, content: str = None) -> str:
        """기사 카테고리 분류"""
        # 간단한 키워드 기반 분류
        keywords = {
            'investment': ['주식', '투자', '펀드', '자산'],
            'banking': ['예금', '적금', '은행', '금리'],
            'loan': ['대출', '담보', '신용', '이자'],
            'insurance': ['보험', '보장', '청구', '가입'],
            'real_estate': ['부동산', '아파트', '전세', '월세'],
            'tech': ['기술', 'IT', '인공지능', '블록체인'],
            'market': ['시장', '경제', '트렌드', '전망']
        }
        
        text = f"{title} {content if content else ''}"
        
        # 각 카테고리별 키워드 매칭 수 계산
        category_scores = {}
        for category, words in keywords.items():
            score = sum(1 for word in words if word in text)
            category_scores[category] = score
        
        # 가장 높은 점수의 카테고리 반환
        if any(category_scores.values()):
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'  # 기본 카테고리 

    def search_similar_by_vector(self, query_vector: np.ndarray, k: int = 5):
        """벡터 기반 유사 기사 검색"""
        try:
            logger.info(f"벡터 검색 시작 - 벡터 shape: {query_vector.shape}")
            logger.info(f"검색 벡터 타입: {type(query_vector)}, dtype: {query_vector.dtype}")
            
            # numpy 배열을 bytes로 변환
            vector_bytes = query_vector.tobytes()
            
            query = (
                Query(
                    f"(*)=>[KNN {k} @embedding $vec_param AS score]"
                )
                .sort_by("score")
                .return_fields("title", "content", "score")
                .dialect(2)
                .paging(0, k)
            )
            
            results = self.redis.ft("article_idx").search(
                query,
                {
                    "vec_param": vector_bytes
                }
            )
            
            logger.info(f"검색 결과: {len(results.docs)}개")
            for doc in results.docs:
                logger.info(f"문서 ID: {doc.id}, 점수: {doc.score}")
            
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 중 에러 발생: {str(e)}", exc_info=True)
            raise e 

    @lru_cache(maxsize=1000)
    def _get_article_vector(self, article_id: str) -> npt.NDArray[np.float32]:
        """기사 벡터 조회 (캐싱 적용)"""
        try:
            article_key = f"article:{article_id}"
            embedding_bytes = self.redis.hget(article_key, "embedding")
            
            if embedding_bytes is None:
                raise ValueError(f"Article {article_id} not found")
            
            return np.frombuffer(embedding_bytes, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"기사 벡터 조회 중 에러: {str(e)}")
            raise e

    def _calculate_time_weight(self, timestamp: str) -> float:
        """시간 기반 가중치 계산"""
        try:
            if not timestamp:
                return 0.5
            
            # timestamp가 bytes 타입인 경우 디코딩
            if isinstance(timestamp, bytes):
                timestamp = timestamp.decode('utf-8')
            
            time_diff = datetime.now() - datetime.fromisoformat(timestamp)
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
                            "FLAT",
                            {
                                "TYPE": "FLOAT32",
                                "DIM": self.VECTOR_DIM,
                                "DISTANCE_METRIC": "COSINE"
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