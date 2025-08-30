import os
import uvicorn
import google.generativeai as genai
import json
import numpy as np
import neo4j
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from pydantic import BaseModel 
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# Google API 키 설정 (Gemini API용)
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Neo4j 환경 변수 설정
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("Neo4j 환경 변수가 모두 설정되지 않았습니다.")

# 전역 변수 설정
embedding_model = None
neo4j_driver = None
MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 시 이벤트를 처리하는 컨텍스트 매니저."""
    global embedding_model, neo4j_driver
    print("임베딩 모델을 로드합니다...")
    try:
        embedding_model = SentenceTransformer(MODEL_NAME)
        print("임베딩 모델 로드 성공!")
    except Exception as e:
        print(f"임베딩 모델 로드 실패: {e}")
        raise HTTPException(status_code=500, detail="임베딩 모델 로드에 실패했습니다.")

    print("Neo4j 드라이버를 생성합니다...")
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        print("Neo4j 연결 성공!")
        with neo4j_driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Keyword) REQUIRE n.name IS UNIQUE")
    except Exception as e:
        print(f"Neo4j 연결 실패: {e}")
        raise HTTPException(status_code=500, detail="Neo4j 연결에 실패했습니다.")
    
    yield

    if neo4j_driver:
        neo4j_driver.close()
        print("Neo4j 연결 종료.")

app = FastAPI(title="마인드맵 데이터 파이프라인 (Neo4j)", lifespan=lifespan)

# --- Pydantic 모델 정의 ---
class QuestionRequest(BaseModel):
    question: str

class NodeDetailResponse(BaseModel):
    name: str
    description: str | None

# --- Gemini API 호출 함수들 ---
async def get_direct_answer_from_question(question: str) -> str:
    """Gemini API를 사용하여 사용자의 질문에 직접 답변합니다."""
    prompt = f"다음 질문에 대해 명확하고 상세하게 한국어로 답변해주세요: {question}"
    try:
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"직접 답변 생성 중 오류 발생: {e}")
        return "답변을 생성하는 데 실패했습니다."

async def extract_keywords_from_question(question: str) -> List[str]:
    """Gemini API를 사용하여 질문에서 핵심 키워드를 추출합니다."""
    prompt = (
        f"You are a highly specialized keyword extraction bot. Your sole function is to identify and extract core, foundational learning keywords from a given text.\n"
        f"Your output is strictly limited to a clean, comma-separated list of keywords. No other text, explanations, or punctuation are permitted.\n"
        f"**Extract keywords that are explicitly present in the provided text. Do not add any keywords that are not directly mentioned in the input text.**\n"
        f"Example 1:\nQuestion: TCP/IP가 뭐야? 설명해줄래?\nOutput: TCP, IP\n"
        f"Example 2:\nQuestion: 마케팅에서 4P 전략이 뭔지 알려줘\nOutput: 마케팅, 4P 전략\n"
        f"Question: {question}\nOutput:"
    )
    try:
        response = await model.generate_content_async(prompt)
        return [keyword.strip() for keyword in response.text.split(',') if keyword.strip()]
    except Exception as e:
        print(f"키워드 추출 중 오류 발생: {e}")
        return []

async def get_additional_info_and_questions(keywords: List[str]) -> Dict[str, Any]:
    """Gemini API를 사용하여 '새로운' 키워드에 대한 설명과 확장 질문을 생성합니다."""
    if not keywords:
        return {"answers": [], "expansion_questions": []}
    
    prompt = (
        f"Based on the following keywords: {', '.join(keywords)}, provide a concise explanation for each and generate related, follow-up questions to encourage further learning.\n"
        f"Format your response as a JSON object with two keys: 'answers' and 'expansion_questions'.\n"
        f"1. 'answers': A list of objects, each with a 'keyword' and a 'description'.\n"
        f"2. 'expansion_questions': A list containing a minimum of 1 and a maximum of 3 thought-provoking questions.\n\n"
        f"**IMPORTANT: All descriptions and questions must be written in Korean.**\n\n"
        f"Example:\n"
        f"{{\n"
        f"  \"answers\": [\n"
        f"    {{\"keyword\": \"TCP\", \"description\": \"전송 제어 프로토콜으로, 신뢰성 있는 데이터 전송을 보장합니다.\"}},\n"
        f"    {{\"keyword\": \"UDP\", \"description\": \"사용자 데이터그램 프로토콜으로, 빠른 속도를 우선시합니다.\"}}\n"
        f"  ],\n"
        f"  \"expansion_questions\": [\n"
        f"    \"TCP와 UDP의 동작 기반인 OSI 7계층에 대해 설명해볼 수 있나요?\",\n"
        f"    \"TCP의 흐름 제어와 혼잡 제어 메커니즘은 어떻게 동작하나요?\"\n"
        f"  ]\n"
        f"}}\n\n"
        f"Your output MUST be a valid JSON object. No extra text or explanations.\n"
        f"Keywords for generation: {', '.join(keywords)}\n"
        f"JSON output:"
    )
    try:
        response = await model.generate_content_async(prompt)
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"새 키워드 정보 생성 중 오류 발생: {e}")
        return {"answers": [], "expansion_questions": []}

# --- 데이터베이스 및 로직 함수들 ---
async def get_all_keywords_from_db() -> Dict[str, str]:
    """Neo4j DB에서 현재 저장된 모든 키워드를 {이름: 설명} 형태의 딕셔너리로 가져옵니다."""
    with neo4j_driver.session() as session:
        result = session.run("MATCH (n:Keyword) RETURN n.name AS name, n.description AS description")
        return {record["name"]: record["description"] for record in result}

def find_strongest_connections(all_keywords: List[str], embeddings: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
    """모든 키워드에 대해 유사도를 계산하고, 사이클이 없는 최강의 연결만 반환합니다."""
    class UnionFind:
        def __init__(self, items):
            self.parent = {item: item for item in items}
        def find(self, item):
            if self.parent[item] == item: return item
            self.parent[item] = self.find(self.parent[item])
            return self.parent[item]
        def union(self, item1, item2):
            root1, root2 = self.find(item1), self.find(item2)
            if root1 != root2:
                self.parent[root2] = root1
                return True
            return False

    similarity_matrix = cosine_similarity(embeddings)
    all_connections = []
    for i in range(len(all_keywords)):
        for j in range(i + 1, len(all_keywords)):
            if (similarity := similarity_matrix[i, j]) >= threshold:
                all_connections.append({"source": all_keywords[i], "target": all_keywords[j], "similarity": float(similarity)})
    
    all_connections.sort(key=lambda x: x['similarity'], reverse=True)
    uf = UnionFind(all_keywords)
    return [conn for conn in all_connections if uf.union(conn['source'], conn['target'])]

async def save_to_neo4j(new_keywords_with_desc: List[Dict[str, str]], links: List[Dict[str, Any]]):
    """새로운 키워드(노드)와 링크(관계)를 Neo4j에 저장합니다."""
    with neo4j_driver.session() as session:
        for kw_data in new_keywords_with_desc:
            session.run("MERGE (k:Keyword {name: $name}) ON CREATE SET k.description = $description",
                        name=kw_data['keyword'], description=kw_data['description'])
        for link in links:
            session.run("MATCH (a:Keyword {name: $source}) MATCH (b:Keyword {name: $target}) "
                        "MERGE (a)-[r:RELATED_TO]-(b) SET r.similarity = $similarity",
                        source=link['source'], target=link['target'], similarity=link['similarity'])
        print("Neo4j에 데이터 저장 완료!")

# --- FastAPI 엔드포인트 ---
@app.post("/generate_mind_map_data")
async def process_and_generate_mind_map_neo4j(request: QuestionRequest):
    """사용자의 질문을 바탕으로 마인드맵 데이터를 생성하고 Neo4j에 저장합니다."""
    # 1. 사용자의 질문에 대한 직접적인 답변을 먼저 생성합니다.
    direct_answer = await get_direct_answer_from_question(request.question)

    # 2. 질문에서 핵심 키워드를 추출합니다.
    extracted_keywords = await extract_keywords_from_question(request.question)
    if not extracted_keywords:
        raise HTTPException(status_code=400, detail="키워드를 추출할 수 없습니다.")

    # 3. 기존 키워드를 DB에서 가져오고, 새로운 키워드를 필터링합니다.
    all_keywords_in_db = await get_all_keywords_from_db()
    existing_keyword_names = set(all_keywords_in_db.keys())
    new_keywords_to_process = [kw for kw in extracted_keywords if kw not in existing_keyword_names]

    # 4. '새로운' 키워드가 있을 경우에만, 해당 키워드에 대한 정보(설명, 확장 질문)를 생성합니다.
    new_keyword_info = await get_additional_info_and_questions(new_keywords_to_process)

    # 5. 마인드맵 구성을 위해 모든 키워드 목록과 임베딩을 준비합니다.
    all_unique_keywords = list(existing_keyword_names.union(set(extracted_keywords)))
    all_embeddings = embedding_model.encode(all_unique_keywords)
    
    # 6. 키워드 간의 연결(링크)을 생성하고 DB에 저장합니다.
    links_to_save = find_strongest_connections(all_unique_keywords, all_embeddings, 0.4)
    await save_to_neo4j(new_keyword_info.get('answers', []), links_to_save)
    
    # 7. 최종 반환할 응답을 조립합니다.
    full_answer = ""
    if direct_answer and direct_answer != "답변을 생성하는 데 실패했습니다.":
        full_answer += f"{direct_answer}\n\n---\n\n"

    # 새로 생성된 키워드 설명을 전체 목록에 업데이트
    for answer in new_keyword_info.get('answers', []):
        all_keywords_in_db[answer['keyword']] = answer['description']

    if new_keyword_info.get('answers'):
        full_answer += "### 새롭게 추가된 키워드 설명\n\n"
        for ans in new_keyword_info['answers']:
            full_answer += f"**{ans['keyword']}:** {ans['description']}\n\n"
    
    expansion_questions = new_keyword_info.get('expansion_questions', [])
    if expansion_questions:
        full_answer += "\n---\n\n"
        full_answer += "### 더 공부해볼까요? 🤔\n\n"
        for q in expansion_questions[:3]:
            full_answer += f"- {q}\n"
    
    final_nodes = [
        {"id": kw, "name": kw, "description": all_keywords_in_db.get(kw, "설명 없음")}
        for kw in all_unique_keywords
    ]

    final_response = {
        "answer": full_answer.strip(),
        "nodes": final_nodes,
        "links": links_to_save
    }
    
    print("마인드맵 데이터 생성 및 저장 완료. 클라이언트에게 반환합니다.")
    return final_response

@app.get("/node/{node_name}", response_model=NodeDetailResponse)
async def get_node_details(node_name: str = Path(..., description="정보를 조회할 노드의 이름")):
    """
    특정 노드의 이름과 설명을 반환합니다.
    """
    # 관련 노드를 찾는 쿼리 부분을 제거
    query = """
    MATCH (n:Keyword {name: $name})
    RETURN n.name AS name, n.description AS description
    """
    try:
        with neo4j_driver.session() as session:
            result = session.run(query, name=node_name).single()
            if not result:
                raise HTTPException(status_code=404, detail="해당 이름의 노드를 찾을 수 없습니다.")

            # 반환값에서 related_nodes 제거
            return {
                "name": result["name"],
                "description": result["description"]
            }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"노드 상세 정보 조회 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)