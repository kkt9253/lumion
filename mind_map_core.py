import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Set
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# FastAPI 애플리케이션 생성
app = FastAPI(title="마인드맵 핵심 로직 테스트 API")

# ChromaDB 클라이언트와 컬렉션 변수
# 클라이언트는 서버가 시작될 때 단 한 번만 생성됩니다.
client = None
collection = None

# Sentence Transformer 모델 로드
# 이 모델은 키워드를 임베딩 벡터로 변환하는 데 사용됩니다.
embedding_model = None
# 사용할 모델의 이름입니다.
MODEL_NAME = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'

# Lifespan 이벤트 핸들러
# on_event는 사용 중단되었으므로, lifespan을 사용하여 애플리케이션의 시작과 종료를 관리합니다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 및 종료 시 이벤트를 처리하는 컨텍스트 매니저.
    """
    global embedding_model, client, collection

    # 1. 애플리케이션 시작 (startup)
    # 임베딩 모델 로드 (서버 시작 시 한 번만)
    print("임베딩 모델을 로드합니다. (최초 실행 시 다운로드 필요)...")
    try:
        embedding_model = SentenceTransformer(MODEL_NAME)
        print("임베딩 모델 로드 성공!")
    except Exception as e:
        print(f"임베딩 모델 로드 실패: {e}")
        embedding_model = None
        raise HTTPException(status_code=500, detail="임베딩 모델 로드에 실패했습니다.")

    # ChromaDB 클라이언트 생성 및 컬렉션 준비
    print("ChromaDB 클라이언트를 생성하고 초기 데이터를 준비합니다...")
    client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_or_create_collection(name="mind_map_keywords", embedding_function=embedding_function)

    # 150개의 임의 키워드 데이터 생성 및 저장
    keywords = [
        # 컴퓨터 과학 및 IT
        "알고리즘", "자료구조", "운영체제", "네트워크", "데이터베이스", "컴파일러", "파이썬", "자바", "C언어", "REST API",
        "클라우드", "가상화", "컨테이너", "도커", "쿠버네티스", "마이크로서비스", "MSA", "머신러닝", "딥러닝", "인공지능",
        "빅데이터", "Hadoop", "Spark", "NoSQL", "SQL", "프론트엔드", "백엔드", "JavaScript", "React", "Angular",
        "Node.js", "서버", "클라이언트", "웹개발", "HTTP", "HTTPS", "TCP", "IP", "UDP", "OSI 7계층",
        # 경제 및 경영
        "경제학", "경영학", "마케팅", "재무관리", "회계", "시장경제", "수요와 공급", "인플레이션", "환율", "주식",
        "채권", "펀드", "재무제표", "손익계산서", "4P 전략", "브랜드", "고객", "경쟁우위", "SWOT 분석", "블록체인",
        # 사회 및 문화
        "심리학", "사회학", "철학", "역사", "문학", "예술", "문화", "대중문화", "인류학", "정치학",
        "민주주의", "자유주의", "사회주의", "공산주의", "인권", "법학", "윤리", "지속가능성", "환경보호", "지구온난화",
        # 과학 및 기술
        "물리학", "화학", "생물학", "지구과학", "우주론", "상대성이론", "양자역학", "DNA", "유전자", "세포",
        "광합성", "에너지", "전기", "자기", "반도체", "신소재", "나노기술", "로봇공학", "인터넷", "스마트폰",
        # 생활 및 기타
        "요리", "음악", "영화", "스포츠", "건강", "영양", "운동", "수면", "스트레스", "커피",
        "여행", "지도", "기후", "계절", "도시", "건축", "도시계획", "언어", "번역", "인터프리터",
        "교통", "차량", "전기차", "자율주행", "GPS", "공항", "기차", "대중교통", "헬스케어", "금융"
    ]
    
    collection.add(
        documents=keywords,
        ids=keywords
    )
    print("ChromaDB 초기 데이터 저장 완료!")
    
    yield # 이 시점에서 애플리케이션이 요청 처리를 시작합니다.

    # 2. 애플리케이션 종료 (shutdown)
    # 서버가 종료될 때 실행될 코드를 여기에 작성합니다.
    if client:
        # 메모리 기반 클라이언트는 별도 종료 함수가 필요하지 않을 수 있습니다.
        pass

# FastAPI 애플리케이션에 lifespan 핸들러 적용
app = FastAPI(title="마인드맵 핵심 로직 테스트 API", lifespan=lifespan)

# Pydantic 모델: 사용자의 질문을 받기 위한 형식
class QuestionRequest(BaseModel):
    question: str

# --------------------
# 마인드맵 데이터 생성 API 엔드포인트
# --------------------

@app.post("/generate_mind_map_data")
def generate_mind_map_data(request: QuestionRequest):
    """
    사용자의 질문을 바탕으로 ChromaDB에서 관련 키워드를 찾아
    마인드맵 시각화에 필요한 데이터를 생성하여 반환합니다.
    """
    global client, collection, embedding_model

    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB 컬렉션이 초기화되지 않았습니다.")

    print(f"사용자 질문 수신: '{request.question}'")
    
    # 1. 사용자 질문과 가장 유사한 키워드를 VectorDB에서 검색
    # ChromaDB에 질문을 보내 가장 유사한 10개의 키워드를 찾습니다.
    # `query_texts`는 질문 텍스트 자체를 입력하면 내부적으로 임베딩하여 검색합니다.
    search_results = collection.query(
        query_texts=[request.question],
        n_results=10
    )
    
    # 2. 검색된 키워드와 그들의 유사도(거리)를 가져옵니다.
    # search_results의 구조: {'distances': [...], 'documents': [...]}
    searched_keywords = search_results['documents'][0]
    searched_distances = search_results['distances'][0]
    
    # 3. 검색된 키워드들 간의 유사도를 계산하여 노드와 링크를 구성합니다.
    # 이 과정은 검색된 소수의 키워드에 대해서만 수행되므로 매우 효율적입니다.
    
    # 임베딩 모델을 사용하여 검색된 키워드들을 다시 임베딩합니다.
    # ChromaDB에서 가져온 임베딩을 직접 사용해도 되지만, 여기서는 명확성을 위해 다시 인코딩합니다.
    if not embedding_model:
        raise HTTPException(status_code=500, detail="임베딩 모델이 로드되지 않았습니다.")
        
    embeddings_of_searched = embedding_model.encode(searched_keywords)
    similarity_matrix = cosine_similarity(embeddings_of_searched)
    
    # 유사도 임계값 설정
    threshold = 0.55
    
    nodes = [{"id": keyword} for keyword in searched_keywords]
    links = []
    link_set = set() # 중복 링크 방지
    
    for i in range(len(searched_keywords)):
        for j in range(i + 1, len(searched_keywords)):
            similarity = similarity_matrix[i, j]
            
            # 유사도가 임계값을 넘는 경우에만 링크 생성
            if similarity > threshold:
                source_id = searched_keywords[i]
                target_id = searched_keywords[j]
                
                # 링크 정규화: (source, target) 순서를 통일
                link_key = tuple(sorted((source_id, target_id)))
                
                if link_key not in link_set:
                    links.append({
                        "source": source_id,
                        "target": target_id,
                        "similarity": float(similarity)
                    })
                    link_set.add(link_key)
    
    # 4. 모든 노드가 마인드맵에 나타나도록 보장
    connected_nodes = {link['source'] for link in links} | {link['target'] for link in links}
    unconnected_nodes = set(searched_keywords) - connected_nodes

    for node_id in unconnected_nodes:
        links.append({
            "source": node_id,
            "target": node_id,
            "similarity": 1.0 # 자체 연결은 유사도 1로 설정
        })
    
    print(f"\n최종 마인드맵 데이터 생성 완료.")
    print(f"총 {len(nodes)}개의 노드와 {len(links)}개의 링크가 생성되었습니다.")
    
    return {"nodes": nodes, "links": links}

# 서버 실행
if __name__ == "__main__":
    uvicorn.run("mind_map_core:app", host="0.0.0.0", port=8000, reload=True)
