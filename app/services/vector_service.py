from redis.commands.search.field import TextField, VectorField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import numpy as np
from datetime import datetime
from typing import List, Dict

class VectorService:
    def __init__(self, redis_client, bert_model):
        self.redis = redis_client
        self.bert = bert_model
        self.VECTOR_DIM = 768
        self._create_index()
        
        # A/B 테스트를 위한 설정
        self.ab_test_groups = {
            'A': {'content': 0.4, 'cf': 0.3, 'time': 0.2, 'diversity': 0.1},
            'B': {'content': 0.3, 'cf': 0.4, 'time': 0.2, 'diversity': 0.1}
        }
        
        # 컨텍스트 가중치
        self.context_weights = {
            'morning': {'investment': 1.2, 'market_news': 1.1},
            'afternoon': {'loan': 1.2, 'credit': 1.1},
            'evening': {'savings': 1.2, 'insurance': 1.1}
        }

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
        embedding = self.bert.get_embedding(content)
        self.redis.hset(
            f"article:{article_id}",
            mapping={
                "title": title,
                "content": content,
                "embedding": embedding.tobytes()
            }
        )
    
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
    
    def record_user_action(self, user_id, article_id, action_type):
        """사용자 행동 기록 (조회, 좋아요, 공유 등)"""
        timestamp = datetime.now().isoformat()
        self.redis.hset(
            f"user_action:{user_id}:{article_id}:{timestamp}",
            mapping={
                "user_id": user_id,
                "article_id": article_id,
                "action_type": action_type,
                "timestamp": timestamp
            }
        )
    
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
    
    def get_recommendations(self, user_id: str, context: dict, k: int = 10) -> List[Dict]:
        """개선된 하이브리드 추천 시스템"""
        
        # 1. 세션 기반 관심사 파악
        session_interests = self._get_session_interests(user_id)
        
        # 2. 컨텍스트 정보 처리
        time_of_day = self._get_time_of_day()
        context_weights = self.context_weights.get(time_of_day, {})
        
        # 3. A/B 테스트 그룹 할당
        ab_group = self._get_ab_test_group(user_id)
        weights = self.ab_test_groups[ab_group]
        
        # 4. 기본 추천 획득
        content_recs = self.get_diverse_recommendations(user_id, k=k)
        cf_recs = self._get_cf_recommendations(self._find_similar_users(user_id))
        
        # 5. 최종 점수 계산
        final_recs = []
        seen_articles = set()
        
        for content_rec in content_recs:
            article_id = content_rec['article_id']
            if article_id in seen_articles:
                continue
                
            # 기본 점수 계산
            base_score = self._calculate_base_score(
                content_rec, 
                cf_recs, 
                weights,
                session_interests
            )
            
            # 컨텍스트 가중치 적용
            category = content_rec['category']
            context_multiplier = context_weights.get(category, 1.0)
            
            # 최종 점수
            final_score = base_score * context_multiplier
            
            final_recs.append({
                **content_rec,
                'final_score': final_score,
                'ab_group': ab_group
            })
            seen_articles.add(article_id)
        
        # 6. 결과 로깅
        self._log_recommendation_results(user_id, final_recs, ab_group)
        
        return sorted(final_recs, key=lambda x: x['final_score'], reverse=True)[:k]

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

    def _log_recommendation_results(self, user_id: str, recommendations: List[Dict], 
                                  ab_group: str):
        """추천 결과 로깅"""
        timestamp = datetime.now().isoformat()
        log_data = {
            'user_id': user_id,
            'timestamp': timestamp,
            'ab_group': ab_group,
            'recommendations': [
                {
                    'article_id': rec['article_id'],
                    'score': rec['final_score'],
                    'category': rec['category']
                } for rec in recommendations
            ]
        }
        
        self.redis.hset(
            f"recommendation_log:{user_id}:{timestamp}",
            mapping=log_data
        )

    def _get_ab_test_group(self, user_id: str) -> str:
        """A/B 테스트 그룹 할당"""
        if not self.redis.exists(f"ab_test_group:{user_id}"):
            group = 'A' if hash(user_id) % 2 == 0 else 'B'
            self.redis.set(f"ab_test_group:{user_id}", group)
        return self.redis.get(f"ab_test_group:{user_id}")

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