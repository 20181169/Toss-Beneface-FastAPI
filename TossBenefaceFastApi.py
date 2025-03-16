import uvicorn
import os
import numpy as np
import cv2
import face_recognition
import requests
import pymysql
import joblib
import io
import re
import time
import random

# Google Cloud Vision 관련 라이브러리
from google.cloud import vision
from google.oauth2 import service_account

from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI  # OpenAI 라이브러리 사용 (환경 변수 OPENAI_API_KEY 설정 필요)
from typing import List, Optional
import pandas as pd

# SQLAlchemy 관련 라이브러리 추가
#from sqlalchemy import create_engine
#from sqlalchemy.orm import sessionmaker





# ------------------------------------------------
# FastAPI 인스턴스 생성 및 CORS 설정
# ------------------------------------------------
app = FastAPI()

# CORS 설정: http://localhost:3000 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# ------------------------------------------------
# 얼굴 인식 관련 전역 변수 및 모델 로딩
# ------------------------------------------------
known_faces = []         # 등록된 각 얼굴의 인코딩(embedding) 리스트
known_member_ids = []    # 등록된 얼굴에 대응하는 memberId

# 앱 시작 시 얼굴 데이터를 미리 로드
@app.on_event("startup")
def startup_event():
    load_faces_from_api()

def load_faces_from_api():
    """
    Spring Boot API로부터 얼굴 데이터(이미지 URL 및 memberId)를 받아와서,
    각 이미지에서 얼굴 인코딩을 추출하고 전역 변수에 저장합니다.
    """
    global known_faces, known_member_ids

    #api_url = "http://tossbeneface.com:8080/api/faces/all"
    #api_url = "http://52.79.200.48:8080/api/faces/all"
    #api_url = "http://localhost:8080/api/faces/all"
    #api_url = "http://52.79.200.48:8080/api/faces/all"
    api_url = "https://api.tossbeneface.com/api/faces/all"
    response = requests.get(api_url)
    print('all response')
    print(response)
    temp_faces = []
    temp_member_ids = []

    if response.status_code == 200:
        faces_data = response.json()  
        # 예: [{ "id": 1, "memberId": 2, "imageUrl": "..." }, { ... }, ...]
        for face in faces_data:
            member_id = face["memberId"]
            image_url = face["imageUrl"]

            try:
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    image_bytes = img_response.content
                    np_array = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        temp_faces.append(encodings[0])
                        temp_member_ids.append(member_id)
                else:
                    print(f"이미지 다운로드 실패: {img_response.status_code}, URL: {image_url}")
            except Exception as e:
                print(f"이미지 다운로드 중 예외 발생: {e}, URL: {image_url}")
    else:
        print(f"Spring Boot API 요청 실패. 상태 코드: {response.status_code}")

    known_faces = temp_faces
    known_member_ids = temp_member_ids

    print(f"[startup] 제대로된 얼굴은: {len(known_faces)}개")


def get_card_details_by_number(recognized_number: str):
    """
    인식된 카드번호에 해당하는 카드 정보를 CSV 파일에서 조회하는 함수.
    카드번호의 앞 6자리(BIN)를 기준으로 매칭하며,
    일치하는 카드가 여러 개 있을 경우 중복 제거 후 첫번째 정보를 반환합니다.
    """
    try:
        df = load_card_data()
        recognized_number = recognized_number.replace(" ", "")
        user_bin = recognized_number[:6]
       
        # CSV의 'bin' 컬럼은 쉼표로 여러 BIN을 가질 수 있으므로 split 처리
        def parse_bin_list(bin_str):
            return [b.strip().replace(" ", "") for b in str(bin_str).split(",")]
        df["bin_list"] = df["bin"].astype(str).apply(parse_bin_list)
       
        matched = df[df["bin_list"].apply(lambda bins: user_bin in bins)]
        if not matched.empty:
            # 중복 제거: '카드 이미지', '카드명', '법인' 기준
            matched = matched.drop_duplicates(subset=["카드 이미지", "카드명", "법인"])
            row = matched.iloc[0]
            return {
                "image_url": row["카드 이미지"],
                "card_name": row["카드명"],
                "card_company": row["법인"]
            }
    except Exception as e:
        print(f"카드 정보 조회 실패: {e}")
    # 조회 실패 시 기본값 반환
    return {
        "image_url": "https://default-image-url.example.com/default.png",
        "card_name": "[카드명 없음]",
        "card_company": "[카드사 없음]"
    }
    
def load_last_five_faces_from_api():
    global known_faces, known_member_ids

    #api_url = "http://tossbeneface.com:8080/api/faces/last5"
    #api_url = "https://52.79.200.48:8080/api/faces/last5"
    #api_url = "http://localhost:8080/api/faces/last5"
    #api_url = "http://52.79.200.48:8080/api/faces/last5"
    api_url = "https://api.tossbeneface.com/api/faces/last5"
    
    response = requests.get(api_url)
    print('last5 response')
    print(response.text)
    
    # 새로 가져온 5개의 인코딩을 임시로 담을 리스트
    new_faces = []
    new_member_ids = []

    if response.status_code == 200:
        faces_data = response.json()  # 최근 5개의 데이터
        for face in faces_data:
            member_id = face["memberId"]
            image_url = face["imageUrl"]

            try:
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    image_bytes = img_response.content
                    np_array = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        new_faces.append(encodings[0])
                        new_member_ids.append(member_id)
                else:
                    print(f"이미지 다운로드 실패: {img_response.status_code}, URL: {image_url}")
            except Exception as e:
                print(f"이미지 다운로드 중 예외 발생: {e}, URL: {image_url}")
    else:
        print(f"Spring Boot API 요청 실패. 상태 코드: {response.status_code}")

    # [중요] "기존 데이터를 덮어쓰는" 대신, "추가(extend)"를 사용
    known_faces.extend(new_faces)
    known_member_ids.extend(new_member_ids)

    print(f"[refresh] 최근 5개 얼굴 데이터 로드 완료, 정상 얼굴: {len(new_faces)}개를 추가했습니다. "
          f"총 {len(known_faces)}개의 얼굴 인코딩 보유 중.")

@app.post("/fastapi/refresh-faces")
def refresh_faces():
    """
    얼굴 데이터 재로딩 엔드포인트.
    현재 예시에서는 "최근 5개"만 불러오는 메서드를 호출.
    """
    load_last_five_faces_from_api()
    return {"message": "Faces reloaded (last 5)"}

@app.post("/fastapi/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    업로드된 이미지 파일에서 얼굴을 인식하고, 등록된 얼굴과 비교하여 memberId를 반환합니다.
    """
    try:
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"message": "Invalid image file"})

        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        if len(face_encodings) == 0:
            return JSONResponse(
                status_code=200, content={"result": "NoFace", "message": "No face detected"}
            )

        face_encoding = face_encodings[0]
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        if len(face_distances) == 0:
            # 등록된 얼굴이 아예 없으면 처리
            return JSONResponse(
                status_code=200,
                content={"result": "NoMatch", "message": "No faces in DB"},
            )

        # 가장 매칭도가 높은 인덱스 찾기
        best_match_index = np.argmin(face_distances)
        THRESHOLD = 0.3

        recognized_member = None
        if face_distances[best_match_index] < THRESHOLD:
            recognized_member = known_member_ids[best_match_index]
            print(f"[인식 성공] memberId={recognized_member}")
        else:
            print(f"[인식 실패] Threshold({THRESHOLD}) 초과")

        return JSONResponse(
            status_code=200,
            content={"result": "success", "memberId": recognized_member},
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})

# ------------------------------------------------
# 예측 관련 엔드포인트 (/predict)
# ------------------------------------------------
class ModelInput(BaseModel):
    전월실적: float
    결제금액: float
    혜택받은횟수: float
    이번달실적: float
    혜택받은금액: float
    Benefit: float
    limit_once: float
    limit_month: float
    min_pay: float
    min_per: float
    monthly: float

# 모델 로딩 (한 번만 로드)
model_path = r"sql/rf_model_modify.pkl"
with open(model_path, "rb") as f:
    model = joblib.load(f)

@app.post("/fastapi/predict")
async def predict(input_data: ModelInput):
    """
    전달된 입력 데이터를 기반으로 모델 예측을 수행합니다.
    """
    features = np.array([[ 
        input_data.전월실적,
        input_data.결제금액,
        input_data.혜택받은횟수,
        input_data.이번달실적,
        input_data.혜택받은금액,
        input_data.Benefit,
        input_data.limit_once,
        input_data.limit_month,
        input_data.min_pay,
        input_data.min_per,
        input_data.monthly
    ]])
    
    prediction = model.predict(features)
    
    return {"prediction": prediction.tolist()}

# ------------------------------------------------
# 데이터 분석 및 LLM 호출 관련 엔드포인트 (/analyze)
# ------------------------------------------------
# DB 연결 함수
def get_connection():
    return pymysql.connect(
        host='db-aivle-bigproject.c16qqgc0i5qn.ap-northeast-2.rds.amazonaws.com',
        user='dev',
        password='',
        database='toss_beneface',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

class SQLRequest(BaseModel):
    store_params: list
    district_params: list

def read_sql_file(file_path, separator="-- QUERY_SEPARATOR"):
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_content = file.read()
    return sql_content.split(separator)

def fetch_sql_data(connection, sql_file: str, params: list):
    queries = read_sql_file(sql_file)
    results = []
    with connection.cursor() as cursor:
        for i, query in enumerate(queries):
            query = query.strip()
            if query:
                cursor.execute(query, params[i])
                results.append(cursor.fetchall())
    return results

def insert_analysis_history(connection, store_id, analysis_result):
    query = """
    INSERT INTO analysis_history (store_id, analysis_script)
    VALUES (%s, %s)
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (store_id, analysis_result))
        connection.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 기록 저장 오류: {str(e)}")

# OpenAI 관련 설정 및 LLM 호출 함수
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
openai_client = OpenAI(api_key=api_key)

def generate_analysis(store_data, district_data, prediction):
    prompt = f"""
    아래는 가맹점과 상권에 대한 매출 데이터입니다.
    데이터를 분석하여 아래 내용을 포함한 보고서를 작성해주세요:
    - 매출 흐름: 증가, 감소, 변화 패턴 등
    - 주요 매출 요일, 성별, 연령대, 시간대
    - 가맹점과 상권 데이터를 비교한 결과
    - 예측된 매출 흐름
    
    가맹점 데이터:
    {store_data}

    상권 데이터:
    {district_data}

    예측된 데이터:
    {prediction}

    위 데이터를 비교하여 주요 인사이트를 도출하고, 향후 전략을 제안해주세요.
    각각의 분석 내용을 문단으로 나누어 가독성 좋게 작성해주세요.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 오류: {str(e)}")

@app.post("/fastapi/analyze")
async def analyze_data(request: SQLRequest):
    try:
        connection = get_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 연결 실패: {str(e)}")
    
    try:
        store_results = fetch_sql_data(connection, "sql/select_store_data.sql", [(request.store_params[0]) for _ in range(5)])
        district_results = fetch_sql_data(connection, "sql/select_district_data.sql", [(request.district_params[0], request.district_params[1]) for _ in range(6)])
        predicted_results = fetch_sql_data(connection, "sql/select_predicted_district_data.sql", [(request.district_params[0], request.district_params[1]) for _ in range(4)])
        
        store_data = "\n".join([str(result) for result in store_results])
        district_data = "\n".join([str(result) for result in district_results])
        predicted_data = "\n".join([str(result) for result in predicted_results])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL 실행 오류: {str(e)}")
    
    try:
        analysis_result = generate_analysis(store_data, district_data, predicted_data)
        insert_analysis_history(connection, request.store_params[0], analysis_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 오류: {str(e)}")
    finally:
        connection.close()
    
    return {
        "store_data": store_data,
        "district_data": district_data,
        "predicted_data": predicted_data,
        "analysis_result": analysis_result
    }

# ------------------------------------------------
# [추가] 카드 OCR 및 정보 추출 API 엔드포인트 (/api/ocr-card)
# ------------------------------------------------
# Google Cloud Vision 클라이언트 설정 (같은 디렉토리에 있는 JSON 키 파일 사용)
json_key_file = "sql/striped-option-449805-v9-4da92e4186de.json"
credentials = service_account.Credentials.from_service_account_file(json_key_file)
vision_client = vision.ImageAnnotatorClient(credentials=credentials) # 변수명만 client -> vision_client 로 변경 필요
print(type(vision_client))
def detect_card(frame):
    """
    이미지에서 카드와 유사한 사각형(카드)을 감지합니다.
    ROI 내에서 그레이스케일 변환, 가우시안 블러, 엣지 검출 및 적응형 이진화 후 윤곽선을 검출하여
    4각형(사각형) 중 면적 조건에 맞는 것을 카드로 간주합니다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)       # 블러 효과
    edged = cv2.Canny(blurred, 30, 150)                # 엣지 검출

    # 적응형 이진화 처리 (더 정확한 감지를 위해)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)

        # 카드 크기 및 사각형 형태 판별 (면적은 필요에 따라 조정)
        if len(approx) == 4 and 5000 < area < 50000:
            detected = True
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)  # 감지된 카드 강조 표시

    return detected, frame

def perform_ocr_from_bytes(image_bytes):
    """
    이미지 바이트 데이터를 받아 Google Cloud Vision API로 텍스트 인식을 수행합니다.
    """
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    else:
        return ""

def extract_card_info(text):
    """
    OCR 결과 텍스트에서 카드번호, 유효기간(Date), CVC 코드 및 기타 정보를 정규표현식으로 추출합니다.
    """
    # 1️⃣ 카드번호 찾기 (4자리씩 4묶음)
    card_number = re.findall(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', text)
    card_number = "\n".join(card_number) if card_number else "[없음]"

    # 2️⃣ 유효기간(Date) 찾기 (MM/YY, MM/YYYY, MM-YY, MM.YY)
    date_info = re.findall(r'\b\d{2}[\/\.\-]\d{2,4}\b', text)
    date_info = "\n".join(date_info) if date_info else "[없음]"

    # 3️⃣ CVC 코드 찾기 (3자리 또는 4자리)
    cvc_info = re.findall(r'\b\d{3,4}\b', text)
    # 보통 'cvc', 'cvv', 'security' 등의 단어 근처에 위치하는 숫자를 필터링 (필요에 따라 개선 가능)
    cvc_info = [cvc for cvc in cvc_info if "cvc" in text.lower() or "cvv" in text.lower() or "security" in text.lower()]
    cvc_info = "\n".join(cvc_info) if cvc_info else "[없음]"

    # 4️⃣ 기타 정보 (카드번호, 유효기간, CVC 제외한 나머지)
    other_text = re.sub(r'\b(?:\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}|\d{2}[\/\.\-]\d{2,4}|\d{3,4})\b', '', text)
    other_text = other_text.strip() if other_text else "[없음]"

    return card_number, date_info, cvc_info, other_text


class VoiceData(BaseModel):
    text: str
    brand: Optional[str] = None
    menus: Optional[List[str]] = None

def clean_gpt_answer(gpt_answer: str) -> str:
    """
    GPT로부터 받은 응답 문자열에서
    - triple backticks (``` 혹은 ```json)
    - 이스케이프된 줄바꿈 (\\n)
    - 역슬래시(\)
    등을 제거하여 반환.
    """
    # 1) 백틱 제거 (```json, ``` 모두 제거)
    cleaned = re.sub(r"```(?:json)?", "", gpt_answer)  # ```json 또는 ``` 모두 제거
    # 2) 혹시 남은 ```도 제거
    cleaned = cleaned.replace("```", "")
    
    # 3) 이스케이프된 줄바꿈(\\n) 제거
    cleaned = cleaned.replace("\\n", "")
    
    # 4) \"(이스케이프된 쌍따옴표)를 실제 쌍따옴표로 바꾸고 싶다면:
    cleaned = cleaned.replace("\\\"", "\"")
    
    # 5) 남아있는 역슬래시(\) 전부 제거
    cleaned = cleaned.replace("\\", "")
    
    # 6) 앞뒤 공백 정리
    cleaned = cleaned.strip()
    
    return cleaned

@app.post("/fastapi/voice-process")
def process_voice(data: VoiceData):
    print(data)
    user_text = data.text
    brand_name = data.brand
    available_menus = data.menus  # ["아메리카노", "라떼", ...]

    # 1) system 프롬프트 + user 프롬프트 구성
    #    - 사용자가 말한 텍스트(user_text)와 available_menus를 함께 제시해
    #      "가장 유사한 메뉴를 골라줘" 같은 식으로 GPT를 유도
    system_prompt = f"""
    너는 {brand_name} 카페의 음성 주문을 받는 직원이다.
    아래는 현재 주문 가능한 메뉴들의 목록이다:
    {', '.join(available_menus)}

    1. 만약 고객이 주문한 메뉴와 정확히 동일한 메뉴가 있으면 동일한 메뉴를 추천해주면 돼. 예를들어 "~ 메뉴를 주문해 드릴게요"
    2. 만약 고객이 주문한 메뉴와 정확히 동일한 메뉴는 없지만 비슷한 메뉴가 있으면 비슷한 메뉴를 추천해주면 돼. 예를들어 "고객님이 주문한 메뉴는 ~ 이지만 저희 매장에서는 판매하고 있지 않습니다. 비슷한 ~ 메뉴를 주문해 드릴까요?"
    3. 만약 고객이 주문한 메뉴와 비슷한 메뉴도 없으면 보유중인 메뉴들 중에 random한 메뉴를 추천해주면 돼. 예를들어 "고객님이 주문하신 ~ 메뉴는 저희 매장에서 판매하고 있지 않습니다. 가장 있기 있는 메뉴인 ~를 드셔보시겠어요?"
    무조건 메뉴를 하나는 추천해 줘야하고 response 형식은 json 형태로 menu ,assistant_text 형태로 반환해야해.
    """
    user_prompt = user_text

    try:
        # 예: OpenAI ChatCompletion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7
        )

        gpt_answer = response.choices[0].message.content.strip()
        gpt_answer = clean_gpt_answer(gpt_answer)
        return {
            "brand": brand_name,
            "available_menus": available_menus,
            "user_text": user_text,
            "gpt_answer": gpt_answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 오류: {str(e)}")

def extract_card_info(text: str):
    """
    OCR 결과에서 카드번호, 유효기간(Date), CVC를 정규식으로 분리
    """
    # 1️. 카드번호 찾기 (4자리씩 4묶음: 16자리)
    card_number = re.findall(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', text)
    card_number = ["".join(re.findall(r'\d', card)) for card in card_number] # 숫자만 남기도록 변환 (공백 및 하이픈 제거)
    card_number = "\n".join(card_number) if card_number else "[없음]"
 
    # 2️. 유효기간(Date) 찾기 (MM/YY, MM/YYYY, MM-YY, MM.YY 등)
    date_info = re.findall(r'\b\d{2}[\/\.\-]\d{2,4}\b', text)
    # YYYY 형식이 포함된 경우, 뒤의 2자리만 남기고 변환
    date_info = [re.sub(r'(\d{2})(\d{2})$', r'\1\2', date) if len(re.sub(r'[\/\.\-]', '', date)) == 6 else date for date in date_info]
    # 날짜 정보에서 /, -, . 등의 문자를 제거하고 숫자 4자리만 유지
    date_info = [re.sub(r'[\/\.\-]', '', date)[:4] for date in date_info]
    date_info = "\n".join(date_info) if date_info else "[없음]"
 
    # 3️. CVC 코드 찾기 (3~4자리),
    #   "CVC", "CVV", "Security" 같은 단어 근처를 확인하는 로직을 추가할 수도 있지만,
    #   여기서는 단순 숫자 3~4자리로 필터링
    #   (더 엄격하게 하려면 OCR 결과에서 'cvc' 인근을 찾는 방식도 가능)
    cvc = re.findall(r'\b\d{3,4}\b', text)
    # 간단 예: 3자리/4자리 중 마지막 1~2개만 CVC일 가능성이 높음
    # 실제론 맥락 분석이 더 필요할 수 있음.
    cvc_info = cvc[-1] if len(cvc) > 0 else "[없음]"  # 예시로 가장 마지막 매칭을 반환
 
    # 4️. 기타 텍스트 (카드번호, 유효기간, CVC 제외)
    other_text = re.sub(r'\b(?:\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}|\d{2}[\/\.\-]\d{2,4}|\d{3,4})\b', '', text)
    other_text = other_text.strip() if other_text else "[없음]"
 
    return card_number, date_info, cvc_info, other_text
 
@app.post("/fastapi/ocr-card")
async def ocr_card(file: UploadFile = File(...)):
    """
    React에서 FormData로 넘어온 파일(이미지)을 받아
    Google Cloud Vision OCR → 카드 정보 추출 → 실제 카드 정보 조회 후 JSON 반환
    """
    # 파일 내용 읽어들이기
    content = await file.read()
 
    # Google Cloud Vision에 전달할 Image 객체 생성
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image) # client -> vision_client
    texts = response.text_annotations
 
    # OCR 결과 파싱
    if texts:
        extracted_text = texts[0].description  # 전체 OCR 텍스트 (가장 상위 항목)
        card_number, date_info, cvc_info, other_text = extract_card_info(extracted_text)
       
        # 카드번호가 인식되었는지 확인 (단순 로직)
        card_detected = True if card_number != "[없음]" else False
       
        # 조회를 위해 카드번호에서 공백, 줄바꿈 제거
        lookup_number = card_number.replace("\n", "").replace(" ", "")
        card_details = get_card_details_by_number(lookup_number)
       
        response_body = {
            "card_detected": card_detected,
            "card_number": card_number,
            "date_info": date_info,
            "cvc_info": cvc_info,
            "other_text": other_text,
            "card_image_url": card_details.get("image_url", ""),
            "card_name": card_details.get("card_name", ""),
            "card_company": card_details.get("card_company", "")
        }
 
        return response_body
 
    else:
        return {
            "card_detected": False
        }


# @app.post("/api/ocr-card")
# async def ocr_card(file: UploadFile = File(...)):
#     """
#     1. 업로드된 이미지를 서버의 `img/` 디렉토리에 저장.
#     2. OCR 로직 제거하고 더미 데이터 반환.
#     """
#     try:
#         # 이미지 저장 (덮어쓰기 방식)
#         with open(IMAGE_PATH, "wb") as image_file:
#             image_file.write(await file.read())

#         # 로컬 이미지 파일 경로 반환
#         image_url = "https://ai0310bucket.s3.ap-southeast-2.amazonaws.com/kpass.png"  # 프론트에서 접근 가능하도록 URL 반환

#         # 더미 데이터 응답
#         return JSONResponse(content={
#             "card_detected": True,
#             "card_number": "0000000000000000",
#             # "date_info": "null",
#             "cvc_info": "000",
#             "card_image_url": image_url,  # 저장된 이미지 경로 반환
#             "card_name": "K pass 신한카드",
#             "card_company": "신한카드"
#         })

#     except Exception as e:
#         return HTTPException(status_code=500, detail=str(e))


# MySQL 연결 함수
def get_connection():
    """
    MySQL 데이터베이스 연결
    """
    return pymysql.connect(
        host='db-aivle-bigproject.c16qqgc0i5qn.ap-northeast-2.rds.amazonaws.com',
        user='dev',
        password='',
        database='toss_beneface',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# MySQL 연결 함수
def get_connection_local():
    """
    MySQL 데이터베이스 연결
    """
    return pymysql.connect(
        host='127.0.0.1',
        user='yeji',
        password='',
        database='mydb',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# SQL 요청 데이터 구조
class SQLRequest(BaseModel):
    store_params: list
    district_params: list
    
# 카드 정보 요청 모델
class CardInfo(BaseModel):
    cardName: str
    cardNumber: str
    cardCompany: str
    expiry: str
    cvc: str
    password: str
    
# 카드 데이터 모델
class Card(BaseModel):
    cardName: str
    cardCompany: str
    cardImage: str
    amount: int

def read_sql_file(file_path, separator="-- QUERY_SEPARATOR"):
    """
    SQL 파일을 읽어 구분자로 나누어 쿼리 리스트를 반환
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_content = file.read()
    return sql_content.split(separator)

def fetch_sql_data(connection, sql_file: str, params: list):
    """
    SQL 파일에서 분리된 쿼리를 실행하여 결과를 반환
    """
    queries = read_sql_file(sql_file)
    results = []

    with connection.cursor() as cursor:
        for i, query in enumerate(queries):
            query = query.strip()
            if query:
                cursor.execute(query, params[i])  # 각 쿼리와 대응하는 파라미터 전달
                results.append(cursor.fetchall())

    return results

# @app.post("/api/save-card")
# async def save_card(card: CardInfo):
#     """
#     카드 정보를 MySQL 데이터베이스에 저장하는 엔드포인트
#     """
#     try:
#         connection = get_connection_local()
#         now_per = random.randint(0, 90000) * 10
#         fetch_sql_data(connection, "sql/insert_card_info.sql", 
#                        [(card.cardNumber, card.cvc, card.password, card.expiry, 1, now_per, card.cardName, card.cardCompany)])
#         connection.commit()

#         # 존재하는 카드인지 확인 필요!!!!!!!!!!!!
#         # 현재 없는 카드면 join이 안되기 때문에 오류 없이 그냥 저장 안 됨!!!!!
#         return {"message": "카드 정보 저장 성공"}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"카드 저장 실패: {str(e)}")

#     finally:
#         connection.close()

@app.get("/fastapi/cards", response_model=List[Card])
async def get_cards(member_id: int):
    """
    사용자의 카드 데이터를 반환하는 API
    Args:
        member_id (int): 사용자의 고유 ID
    Returns:
        List[Card]: 카드 목록
    """
    try:
        # 데이터베이스 연결
        connection = get_connection_local()
        raw_result = fetch_sql_data(connection, "sql/select_card_list.sql", [(member_id)])
        
        if not raw_result:
            raise HTTPException(status_code=404, detail="No cards found for the user")

        result = [Card(**row) for row in raw_result[0]]
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    finally:
        if connection:
            connection.close()
            
@app.post("/fastapi/save-card")
async def save_card(card: CardInfo):
    """
    카드 정보를 MySQL 데이터베이스에 저장하는 엔드포인트
    """
    try:
        connection = get_connection_local()
        cursor = connection.cursor()

        # 1️. 카드가 존재하는지 먼저 확인
        cursor.execute("""
            SELECT id FROM card 
            WHERE card_name = %s AND card_company = %s
        """, (card.cardName, card.cardCompany))
        
        card_data = cursor.fetchone()

        # 2. 존재하지 않는 카드라면 오류 반환
        if not card_data:
            raise HTTPException(status_code=400, detail="존재하지 않는 카드입니다.")

        # 3️. 카드가 존재하면 정상적으로 저장
        now_per = random.randint(0, 90000) * 10
        last_per = random.randint(int(now_per/10), 100000) * 10
        monthly_split = random.randint(0, 3)
        accrue_benefit = random.randint(100, 4000) * 10
        fetch_sql_data(connection, "sql/insert_card_info.sql", 
                       [(card.cardNumber, card.cvc, card.password, card.expiry, 1, now_per, 
                         last_per, monthly_split, accrue_benefit, card.cardName, card.cardCompany)])
        connection.commit()

        return {"message": "카드 정보 저장 성공"}

    except HTTPException as e:
        raise e  # HTTPException은 그대로 전달

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"카드 저장 실패: {str(e)}")

    finally:
        connection.close()
CSV_PATH = r"sql/cardData_image.csv"

# CSV 데이터 로드 함수
def load_card_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("CSV 파일을 찾을 수 없습니다.")
    return pd.read_csv(CSV_PATH)

@app.get("/fastapi/get-card-image")
async def get_card_image(card_number: str = Query(..., min_length=6)):
    """
    1) 카드번호(card_number) 앞 8자리로 BIN 매칭 우선 시도
    2) 없으면 6자리로 다시 매칭
    3) bin 컬럼이 쉼표(,)로 여러 BIN을 가질 수 있으므로 split 처리
    4) 중복 카드(이미지/카드명/법인)는 drop_duplicates()로 제거
    """
    try:
        df = load_card_data()
 
        ############################################
        # 1) 사용자 BIN (공백 제거)
        ############################################
        # 예: "5361 48 1234" -> "5361481234"
        user_card_number = card_number.replace(" ", "")
 
        ############################################
        # 2) CSV에서 bin 컬럼을 쉼표 split → bin_list
        ############################################
        def parse_bin_list(bin_str):
            return [b.strip().replace(" ", "") for b in str(bin_str).split(",")]
       
        df["bin_list"] = df["bin"].astype(str).apply(parse_bin_list)
 
        ############################################
        # 3) 8자리 매칭 → 없으면 6자리 매칭
        ############################################
        user_bin_8 = user_card_number[:8]
        matched_8 = df[df["bin_list"].apply(lambda bins: user_bin_8 in bins)]
 
        if not matched_8.empty:
            matched_cards = matched_8
        else:
            user_bin_6 = user_card_number[:6]
            matched_6 = df[df["bin_list"].apply(lambda bins: user_bin_6 in bins)]
            matched_cards = matched_6
 
        if matched_cards.empty:
            return {
                "message": "카드 이미지 없음",
                "card_count": 0,
                "cards": []
            }
 
        ############################################
        # 4) 중복 제거
        ############################################
        columns_to_return = ["카드 이미지", "카드명", "법인"]
        matched_subset = matched_cards[columns_to_return]
        matched_no_dup = matched_subset.drop_duplicates(subset=columns_to_return)
 
        ############################################
        # 5) dict 변환 후 반환
        ############################################
        card_list = matched_no_dup.to_dict(orient="records")
 
        return {
            "message": f"{len(card_list)}개의 카드가 매칭되었습니다.",
            "card_count": len(card_list),
            "cards": card_list
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ------------------------------------------------
# 애플리케이션 실행
# ------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)