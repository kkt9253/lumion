import os
import uvicorn
import google.generativeai as genai
import json
import chromadb
import mysql.connector
import numpy as np
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Set
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from datetime import datetime
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
# API 키와 같은 민감 정보는 소스 코드에 직접 노출하지 않고 환경 변수로 관리합니다.
load_dotenv()

# Google API 키 설정 (Gemini API용)
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# MySQL 환경 변수 설정
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]):
    raise ValueError("MySQL 환경 변수가 모두 설정되지 않았습니다.")

# FastAPI 애플리케이션 생성
app = FastAPI(title="마인드맵 데이터 파이프라인")

# ChromaDB 클라이언트와 컬렉션 변수
client = None
collection = None

# Sentence Transformer 모델 로드
embedding_model = None
MODEL_NAME = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'

# MySQL 연결 변수
mysql_conn = None

def create_mysql_tables(cursor):
    """
    MySQL에 필요한 테이블을 자동으로 생성합니다.
    """
    try:
        cursor.execute("USE {}".format(MYSQL_DB))
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id VARCHAR(255) PRIMARY KEY,
                label TEXT NOT NULL,
                created_at DATETIME NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id VARCHAR(255) NOT NULL,
                target_id VARCHAR(255) NOT NULL,
                similarity FLOAT NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
        """)
        print("MySQL 테이블 생성/확인 완료.")
    except mysql.connector.Error as e:
        print(f"MySQL 테이블 생성 중 오류 발생: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 및 종료 시 이벤트를 처리하는 컨텍스트 매니저.
    """
    global embedding_model, client, collection, mysql_conn

    # 1. 애플리케이션 시작 (startup)
    print("임베딩 모델을 로드합니다. (최초 실행 시 다운로드 필요)...")
    try:
        embedding_model = SentenceTransformer(MODEL_NAME)
        print("임베딩 모델 로드 성공!")
    except Exception as e:
        print(f"임베딩 모델 로드 실패: {e}")
        embedding_model = None
        raise HTTPException(status_code=500, detail="임베딩 모델 로드에 실패했습니다.")

    print("ChromaDB와 MySQL 클라이언트를 생성합니다...")
    # ChromaDB 클라이언트 (메모리 내)
    client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_or_create_collection(name="mind_map_keywords", embedding_function=embedding_function)
    
    # MySQL 클라이언트
    try:
        mysql_conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = mysql_conn.cursor()
        create_mysql_tables(cursor)
        cursor.close()
        print("MySQL 연결 성공!")
    except mysql.connector.Error as e:
        print(f"MySQL 연결 실패: {e}")
        mysql_conn = None
        raise HTTPException(status_code=500, detail="MySQL 연결에 실패했습니다.")
    
    yield

    # 2. 애플리케이션 종료 (shutdown)
    if mysql_conn and mysql_conn.is_connected():
        mysql_conn.close()
        print("MySQL 연결 종료.")
    if client:
        # 메모리 기반 ChromaDB 클라이언트는 별도 종료 함수가 필요하지 않을 수 있습니다.
        pass

app = FastAPI(title="마인드맵 데이터 파이프라인", lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str

class ClearDBRequest(BaseModel):
    confirm: bool

async def extract_keywords(question: str) -> List[str]:
    """Gemini API를 사용하여 질문에서 핵심 키워드를 추출합니다."""
    prompt = (
        f"You are a highly specialized keyword extraction bot. Your sole function is to identify and extract core, foundational learning keywords from a given text.\n"
        f"Your output is strictly limited to a clean, comma-separated list of keywords. No other text, explanations, or punctuation are permitted.\n"
        f"To ensure perfect output, follow this internal reasoning process before generating the final output. This process is for your internal use only and must not be included in the final output.\n"
        f"<\ctrl3347>\n"
        f"1. Analyze the user's question. Identify the central subject matter or concept being asked about.\n"
        f"2. Deconstruct the subject. Break down any composite terms into their most fundamental, individual components. For example, 'TCP/IP' becomes 'TCP' and 'IP'.\n"
        f"3. Identify Core Nouns/Concepts. From the deconstructed parts, filter out any words that are not essential learning topics. Exclude meta-phrases like 'What is,' 'explain,' 'tell me about,' or 'describe.'\n"
        f"4. Filter for Relevancy. If the question contains multiple topics, select only the primary ones.\n"
        f"5. Construct the final list. Combine the identified core keywords into a single, clean, comma-separated string. The list must not contain duplicates.\n"
        f"6. Review against rules. Check the final list to ensure it strictly follows the output format.\n"
        f"7. **Extract keywords that are explicitly present in the provided text. Do not add any keywords that are not directly mentioned in the input text.**\n"
        f"<\ctrl3348>\n"
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

def get_existing_keywords_from_db():
    """MySQL DB에서 현재 저장된 모든 키워드를 가져옵니다."""
    cursor = mysql_conn.cursor()
    cursor.execute("SELECT label FROM concepts")
    results = cursor.fetchall()
    cursor.close()
    return [row[0] for row in results]

def find_strongest_connections(all_keywords: List[str], embeddings: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
    """
    모든 키워드에 대해 유사도를 계산하고, 사이클이 없는 최강의 연결만 반환합니다.
    """
    # 유니온-파인드 자료구조를 위한 클래스 (사이클 방지)
    class UnionFind:
        def __init__(self, items):
            self.parent = {item: item for item in items}
        def find(self, item):
            if self.parent[item] == item:
                return item
            self.parent[item] = self.find(self.parent[item])
            return self.parent[item]
        def union(self, item1, item2):
            root1 = self.find(item1)
            root2 = self.find(item2)
            if root1 != root2:
                self.parent[root2] = root1
                return True
            return False

    similarity_matrix = cosine_similarity(embeddings)
    all_connections = []
    # 디버깅용: 모든 키워드 쌍의 유사도를 출력합니다.
    print("\n--- 모든 키워드 쌍의 유사도 ---")
    for i in range(len(all_keywords)):
        for j in range(i + 1, len(all_keywords)):
            similarity = similarity_matrix[i, j]
            print(f"'{all_keywords[i]}' - '{all_keywords[j]}': {similarity:.4f}")
            if similarity >= threshold:
                all_connections.append({
                    "source": all_keywords[i],
                    "target": all_keywords[j],
                    "similarity": float(similarity),
                    "source_idx": i,
                    "target_idx": j
                })
    print("----------------------------\n")
    all_connections.sort(key=lambda x: x['similarity'], reverse=True)
    
    uf = UnionFind(all_keywords)
    links = []
    for conn in all_connections:
        if uf.union(conn['source'], conn['target']):
            links.append(conn)
            
    return links

def save_to_mysql(new_keywords: List[str], links: List[Dict[str, Any]]):
    """
    새로운 키워드(노드)와 링크(엣지)를 MySQL에 저장합니다.
    """
    if not mysql_conn:
        return
        
    cursor = mysql_conn.cursor()
    
    # 1. 새로운 키워드(노드) 저장
    concept_query = "INSERT IGNORE INTO concepts (id, label, created_at) VALUES (%s, %s, %s)"
    concepts_to_insert = [(kw, kw, datetime.now()) for kw in new_keywords]
    cursor.executemany(concept_query, concepts_to_insert)

    # 2. 링크(엣지) 저장
    edge_query = "INSERT IGNORE INTO edges (source_id, target_id, similarity) VALUES (%s, %s, %s)"
    edges_to_insert = [(link['source'], link['target'], link['similarity']) for link in links]
    cursor.executemany(edge_query, edges_to_insert)

    mysql_conn.commit()
    cursor.close()
    print("MySQL에 데이터 저장 완료!")

@app.post("/process_and_generate")
async def process_and_generate_mind_map(request: QuestionRequest):
    """
    사용자의 질문을 바탕으로 마인드맵 데이터를 생성하고 DB에 저장합니다.
    로직: 키워드 추출 -> 중복 확인 -> VectorDB/MySQL 저장 -> 마인드맵 데이터 반환
    """
    global client, collection, embedding_model, mysql_conn

    # 1. 사용자 질문에서 핵심 키워드 추출 (Gemini API)
    new_keywords_raw = await extract_keywords(request.question)
    print(f"추출된 키워드: {new_keywords_raw}")
    if not new_keywords_raw:
        return {"nodes": [], "links": []}

    # 2. ChromaDB에서 중복 키워드 확인 및 분리
    existing_in_chroma_result = collection.get(ids=new_keywords_raw)
    existing_ids_in_chroma = set(existing_in_chroma_result['ids'])
    keywords_to_add_to_chroma = [kw for kw in new_keywords_raw if kw not in existing_ids_in_chroma]
    
    # 3. 신규 키워드 VectorDB에 저장
    if keywords_to_add_to_chroma:
        collection.add(documents=keywords_to_add_to_chroma, ids=keywords_to_add_to_chroma)
        print(f"ChromaDB에 신규 키워드 추가: {keywords_to_add_to_chroma}")

    # 4. 모든 키워드(MySQL + 신규)를 가져와 유사도 측정 준비
    existing_in_mysql = get_existing_keywords_from_db()
    all_unique_keywords = list(set(existing_in_mysql + new_keywords_raw))
    
    all_embeddings = embedding_model.encode(all_unique_keywords)
    
    # 5. 유사도 측정 및 링크 생성 (사이클 방지)
    # 임계값을 0.4로 낮춰서 더 넓은 범위의 연관 관계를 포착합니다.
    links_to_save = find_strongest_connections(all_unique_keywords, all_embeddings, 0.4)
    
    # 6. MySQL에 새로운 노드와 링크 저장
    save_to_mysql(all_unique_keywords, links_to_save)
    
    # 7. 프론트엔드에 반환할 마인드맵 데이터 구성
    nodes_data = [{"id": kw} for kw in all_unique_keywords]
    links_data = [{"source": l['source'], "target": l['target'], "similarity": l['similarity']} for l in links_to_save]
    
    print("마인드맵 데이터 생성 완료. 클라이언트에게 반환합니다.")
    return {"nodes": nodes_data, "links": links_data}

@app.delete("/clear_data")
def clear_all_data(request: ClearDBRequest):
    """
    VectorDB와 MySQL의 모든 데이터를 삭제합니다.
    테스트 용도로 사용됩니다.
    """
    global client, collection, mysql_conn
    if not request.confirm:
        raise HTTPException(status_code=400, detail="데이터 삭제를 위해 'confirm'을 true로 설정해야 합니다.")

    try:
        # ChromaDB 데이터 삭제
        client.delete_collection(name="mind_map_keywords")
        # 컬렉션 재 생성
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        collection = client.get_or_create_collection(name="mind_map_keywords", embedding_function=embedding_function)
        
        # MySQL 데이터 삭제
        if mysql_conn and mysql_conn.is_connected():
            cursor = mysql_conn.cursor()
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM concepts")
            mysql_conn.commit()
            cursor.close()
        
        print("모든 데이터가 성공적으로 삭제되었습니다.")
        return {"message": "모든 데이터가 성공적으로 삭제되었습니다."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 삭제 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("mind_map_core:app", host="0.0.0.0", port=8000, reload=True)
