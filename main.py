import os
import spacy
import dateparser
import datetime
from urllib.parse import quote_plus
from transformers import pipeline

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- [New] Auth & Session Libraries ---
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

# ==============================================================================
# Configuration / 설정
# ==============================================================================
NER_MODEL_DIR = "my_ner_model"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ko-en"

# Hugging Face Secrets에서 열쇠를 가져옵니다.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", "random_secret_string")  # 세션 암호화용

models = {}


# ==============================================================================
# AI Functions (번역, 추출 - 기존과 동일)
# ==============================================================================
def translate_korean_to_english(text):
    is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text)
    if is_korean:
        translated = models["translator"](text, max_length=512)
        return translated[0]['translation_text']
    else:
        return text


def extract_schedule_info(translated_text):
    if not translated_text or not translated_text.strip():
        return "Please enter text.", "", "", ""
    doc = models["nlp"](translated_text)
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
# App Lifecycle & OAuth Setup
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield
    models.clear()
    print("✅ Models cleared.")


app = FastAPI(lifespan=lifespan)

# [New] Session Middleware Configuration (Stores login info)
# [신규] 세션 미들웨어 설정 (로그인 정보 저장)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    # Essential for Hugging Face Spaces (HTTPS environment)
    # Hugging Face Spaces(HTTPS 환경)에서는 필수입니다.
    https_only=True,

    # Allows cookies to be sent when redirecting from Google
    # 구글에서 리다이렉트될 때 쿠키가 전송되도록 허용합니다 ('lax' 권장).
    same_site='lax'
)

# [New] 구글 로그인 설정
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile https://www.googleapis.com/auth/calendar.events'},
)


# --- Pydantic Models ---
class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    original_text: str
    translated_text: str
    date: str
    time: str
    loc: str
    event: str


class AddEventRequest(BaseModel):
    date_str: str
    time_str: str
    loc_str: str
    event_str: str
    description: str


# ==============================================================================
# Endpoints
# ==============================================================================

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.post("/extract", response_model=ExtractResponse)
async def api_extract_schedule(request: ExtractRequest):
    original_text = request.text
    translated_text = translate_korean_to_english(original_text)
    date, time, loc, event = extract_schedule_info(translated_text)
    return ExtractResponse(
        original_text=original_text, translated_text=translated_text,
        date=date, time=time, loc=loc, event=event
    )


# --- [New] Login Endpoints ---

@app.get('/login')
async def login(request: Request):
    # 현재 주소를 기반으로 callback 주소 생성 (https 강제)
    redirect_uri = str(request.url_for('auth')).replace('http://', 'https://')
    # 로컬 테스트용 (localhost일 땐 http 유지)
    if "localhost" in redirect_uri or "127.0.0.1" in redirect_uri:
        redirect_uri = str(request.url_for('auth'))

    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/auth/callback')
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        request.session['user'] = token.get('userinfo')
        request.session['token'] = token
        return RedirectResponse(url='/')
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Login failed: {str(e)}"})


@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    request.session.pop('token', None)
    return RedirectResponse(url='/')


@app.get('/user-info')
async def get_user_info(request: Request):
    user = request.session.get('user')
    return {"user": user}


# --- [New] Add to Calendar Endpoint ---

@app.post("/add-to-calendar")
async def add_to_calendar(request: Request, event_data: AddEventRequest):
    token = request.session.get('token')
    if not token:
        return JSONResponse(status_code=401, content={"error": "Login required"})

    try:
        # 날짜/시간 파싱
        dt_str = f"{event_data.date_str} {event_data.time_str}"
        start_dt = dateparser.parse(dt_str, settings={'PREFER_DATES_FROM': 'future'})
        if not start_dt:
            start_dt = datetime.datetime.now() + datetime.timedelta(hours=1)

        end_dt = start_dt + datetime.timedelta(hours=1)

        # 구글 API에 보낼 데이터
        google_event = {
            'summary': event_data.event_str,
            'location': event_data.loc_str,
            'description': event_data.description,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': 'Asia/Seoul',
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': 'Asia/Seoul',
            },
        }

        # 구글 캘린더 API 호출
        resp = await oauth.google.post(
            'https://www.googleapis.com/calendar/v3/calendars/primary/events',
            json=google_event,
            token=token
        )
        resp.raise_for_status()
        result = resp.json()

        return {"message": "Success", "link": result.get('htmlLink')}

    except Exception as e:
        print(f"Calendar Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)