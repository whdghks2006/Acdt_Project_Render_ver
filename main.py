import spacy
import os
import dateparser
import datetime
from urllib.parse import quote_plus
from transformers import pipeline

# FastAPI와 Pydantic(데이터 검증용) 임포트
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- [ADD] Libraries for serving frontend ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ==============================================================================
# Configuration / 설정
# ==============================================================================
NER_MODEL_DIR = "my_ner_model"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ko-en"

# AI 모델을 저장할 글로벌 변수
models = {}


# ==============================================================================
# 1. (Gradio 코드 재사용) Translation Function / 번역 함수
# ==============================================================================
def translate_korean_to_english(text):
    # (Gradio 앱의 함수와 동일. 그대로 복사)
    is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text)
    if is_korean:
        translated = models["translator"](text, max_length=512)
        return translated[0]['translation_text']
    else:
        return text


# ==============================================================================
# 2. (Gradio 코드 재사용) AI Extraction Function / AI 추출 함수
# ==============================================================================
def extract_schedule_info(translated_text):
    # (Gradio 앱의 함수와 동일. 그대로 복사)
    if not translated_text or not translated_text.strip():
        return "Please enter text.", "", "", ""

    doc = models["nlp"](translated_text)  # nlp 대신 models["nlp"] 사용

    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    times = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
    locs = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    events = [ent.text for ent in doc.ents if ent.label_ == "EVENT"]

    date_str = ", ".join(dates) if dates else "today"
    time_str = ", ".join(times) if times else ""
    loc_str = ", ".join(locs) if locs else ""

    if events:
        event_str = ", ".join(events)
    elif locs:
        event_str = f"Meeting at {loc_str}"
    else:
        event_str = "New Schedule"

    return date_str, time_str, loc_str, event_str


# ==============================================================================
# 3. (Gradio 코드 재사용) Link Generation Function / 링크 생성 함수
# ==============================================================================
def create_calendar_link(date_str, time_str, loc_str, event_str, original_text, translated_text):
    """
    Takes the (manually edited) info and creates a link.
    """
    try:
        # --- ▼▼▼ 이 부분이 누락되었습니다 ▼▼▼ ---
        datetime_text = f"{date_str} {time_str}"
        start_time = dateparser.parse(datetime_text, settings={'PREFER_DATES_FROM': 'future'})

        if start_time is None:
            return None  # 날짜 파싱 실패

        end_time = start_time + datetime.timedelta(hours=1)
        # --- ▲▲▲ 여기까지가 누락된 코드입니다 ▲▲▲ ---

        # 이 변수들이 정의되어야 아래에서 사용할 수 있습니다.
        start_utc = start_time.astimezone(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        end_utc = end_time.astimezone(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')

        # 이 부분(f-string 닫는 중괄호)은 올바르게 수정하셨습니다!
        dates_formatted = f"{start_utc}/{end_utc}"

        details_text = (
            f"🤖 AI-extracted schedule.\n\n"
            f"--- [Original Message] ---\n{original_text}\n\n"
            f"--- [Translated Text] ---\n{translated_text}"
        )

        base_url = "https://www.google.com/calendar/render?action=TEMPLATE"
        url = (
            f"{base_url}"
            f"&text={quote_plus(event_str)}"
            f"&dates={dates_formatted}"
            f"&location={quote_plus(loc_str)}"
            f"&details={quote_plus(details_text)}"
        )
        return url  # URL 문자열만 반환

    except Exception as e:
        print(f"Error creating link: {e}")
        return None  # 실패 시 None 반환


# ==============================================================================
# FastAPI 앱 설정
# ==============================================================================

# 앱 시작 시 모델 로드, 종료 시 메모리 정리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시
    print("🔄 Loading AI Models...")
    if not os.path.exists(NER_MODEL_DIR):
        print(f"❌ Error: NER Model folder not found at {NER_MODEL_DIR}")
        exit()
    try:
        models["nlp"] = spacy.load(NER_MODEL_DIR)
        print("✅ NER Model loaded successfully!")
        models["translator"] = pipeline("translation", model=TRANSLATION_MODEL)
        print("✅ Translation Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        exit()

    yield  # 앱 실행

    # 종료 시
    models.clear()
    print("✅ Models cleared.")


# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)


# --- API 입/출력 데이터 형식 정의 (Pydantic) ---

class ExtractRequest(BaseModel):
    text: str  # 입력 텍스트


class ExtractResponse(BaseModel):
    original_text: str
    translated_text: str
    date: str
    time: str
    loc: str
    event: str


class LinkRequest(BaseModel):
    date_str: str
    time_str: str
    loc_str: str
    event_str: str
    original_text: str
    translated_text: str


class LinkResponse(BaseModel):
    google_calendar_url: str


# --- API 엔드포인트(Endpoint) 정의 ---

# 1. 텍스트 추출 API
@app.post("/extract", response_model=ExtractResponse)
async def api_extract_schedule(request: ExtractRequest):
    original_text = request.text

    # 1. 번역
    translated_text = translate_korean_to_english(original_text)

    # 2. 추출
    date, time, loc, event = extract_schedule_info(translated_text)

    # 3. JSON으로 결과 반환
    return ExtractResponse(
        original_text=original_text,
        translated_text=translated_text,
        date=date,
        time=time,
        loc=loc,
        event=event
    )


# 2. 캘린더 링크 생성 API
@app.post("/generate-link", response_model=LinkResponse)
async def api_generate_link(request: LinkRequest):
    url = create_calendar_link(
        request.date_str,
        request.time_str,
        request.loc_str,
        request.event_str,
        request.original_text,
        request.translated_text
    )

    if url:
        return LinkResponse(google_calendar_url=url)
    else:
        # FastAPI는 자동으로 오류 응답을 생성합니다.
        # 여기서는 간단히 빈 URL로 실패를 알릴 수 있지만,
        # 실제로는 HTTPException을 발생시키는 것이 더 좋습니다.
        return LinkResponse(google_calendar_url="")

# --- [MODIFY] Frontend Serving Configuration ---

# "static" 폴더를 /static URL 경로에 마운트합니다. CSS, JS 파일 접근 허용
# Mount the "static" folder to the /static URL path. Allows access to CSS, JS files.
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 경로 ("/") 접속 시 JSON 대신 static/index.html 파일을 보여줍니다.
# Serve static/index.html instead of JSON when accessing the root path ("/").
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


# (Render 배포를 위한 설정 - Gradio와 동일)
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 8000))  # FastAPI 기본 포트는 8000
    uvicorn.run(app, host="0.0.0.0", port=port)