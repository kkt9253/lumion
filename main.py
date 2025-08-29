import os
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
# API 키와 같은 민감한 정보는 소스 코드에 직접 노출하지 않고 환경 변수로 관리하는 것이 보안에 좋습니다.
# 프로젝트 루트 디렉토리에 .env 파일을 만들고 GOOGLE_API_KEY="YOUR_API_KEY" 형식으로 저장하세요.
load_dotenv()

# Google API 키 설정
# 환경 변수에서 API 키를 가져옵니다. 키가 설정되어 있지 않으면 에러를 발생시켜 프로그램이 즉시 종료되도록 합니다.
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# Gemini API 구성
# genai.configure() 함수를 사용하여 Google Gemini API를 사용하기 위한 초기 설정을 합니다.
genai.configure(api_key=API_KEY)

# 사용할 Gemini 모델 설정
# 여기서는 'gemini-1.5-flash' 모델을 사용합니다. 필요에 따라 다른 모델로 변경할 수 있습니다.
model = genai.GenerativeModel('gemini-2.5-flash')

# FastAPI 애플리케이션 생성
# app 변수를 통해 FastAPI 애플리케이션을 초기화합니다.
app = FastAPI(title="Gemini API 터미널 출력 예제")

# Pydantic 모델 정의
# Postman과 같은 클라이언트로부터 받을 요청의 본문 형식을 정의합니다.
# "question"이라는 이름의 문자열 필드를 반드시 포함해야 합니다.
class QuestionRequest(BaseModel):
    question: str

# 루트 엔드포인트
# 서버가 정상적으로 실행 중인지 확인하기 위한 간단한 API입니다.
@app.get("/")
def read_root():
    return {"message": "Welcome to the Gemini API terminal print example!"}

# 채팅 엔드포인트
# 사용자의 질문을 받아 Gemini API에 전달하고, 응답을 터미널에 출력합니다.
@app.post("/chat")
async def chat_with_gemini(request: QuestionRequest):
    """
    사용자의 질문을 받아 Gemini API에 전달하고, 응답을 터미널에 출력합니다.
    Args:
        request (QuestionRequest): 사용자의 질문을 담고 있는 요청 본문.
    
    Returns:
        dict: 응답이 터미널에 출력되었음을 알리는 메시지.
    """
    try:
        # Gemini API 호출
        # model.generate_content() 함수에 사용자의 질문을 전달하여 AI 응답을 생성합니다.
        print("사용자 질문:", request.question)
        response = model.generate_content(request.question)
        
        # 응답 텍스트를 추출하고 터미널에 출력
        # AI로부터 받은 응답 객체에서 실제 텍스트 내용을 추출합니다.
        answer_text = response.text
        print("-" * 15)
        print("Gemini API 응답:")
        print(answer_text)
        print("-" * 15)

        # 클라이언트에게는 응답이 터미널에 출력되었음을 알리는 메시지를 반환합니다.
        # 실제 응답 내용을 반환하지 않아도 클라이언트가 타임아웃되지 않도록 메시지를 보내줍니다.
        return {"message": "응답이 서버 터미널에 출력되었습니다."}

    except Exception as e:
        # API 호출 중 에러가 발생하면 HTTP 500 에러와 함께 상세 에러 메시지를 반환합니다.
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행
# 이 스크립트를 직접 실행하면 Uvicorn 웹 서버가 시작됩니다.
if __name__ == "__main__":
    # uvicorn.run() 함수를 사용하여 FastAPI 애플리케이션을 실행합니다.
    # "main:app"은 main.py 파일 내의 app 객체를 의미합니다.
    # host="0.0.0.0"은 외부 접속을 허용하며, reload=True는 코드 변경 시 서버를 자동 재시작합니다.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
