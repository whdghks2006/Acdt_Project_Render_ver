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

# --- Auth & Session Libraries ---
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

# ==============================================================================
# Configuration
# ==============================================================================
NER_MODEL_DIR = "my_ner_model"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ko-en"

# Retrieve secrets from environment variables
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", "random_secret_string")

models = {}


# ==============================================================================
# AI Functions
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
# App Lifecycle & Setup
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

# [CRITICAL FIX: Session Middleware]
# same_site='lax': Best for top-level navigation (Direct URL)
# max_age=3600: Forces the cookie to persist for 1 hour
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=True,
    same_site='lax',
    path='/',
    max_age=3600
)

oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile https://www.googleapis.com/auth/calendar.events'},
)


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


@app.get('/login')
async def login(request: Request):
    # Ensure this matches your actual Hugging Face URL exactly
    fixed_redirect_uri = "https://snowmang-ai-scheduler-g14.hf.space/auth/callback"
    return await oauth.google.authorize_redirect(request, fixed_redirect_uri)


@app.get('/auth/callback')
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        print(f"✅ Google Login Success! User: {user_info}")  # Debug Log

        request.session['user'] = user_info
        request.session['token'] = token

        # [FIX] Use 303 See Other for redirect (Better for browser history/cookies)
        return RedirectResponse(url='/', status_code=303)
    except Exception as e:
        print(f"❌ Login Error: {e}")
        return JSONResponse(status_code=400, content={"error": f"Login failed: {str(e)}"})


@app.get('/logout')
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url='/')


@app.get('/user-info')
async def get_user_info(request: Request):
    user = request.session.get('user')
    # Debug: Print to server log if user info is requested but empty
    if not user:
        print("⚠️ User info requested but session is empty!")
    else:
        print(f"ℹ️ User info found for: {user.get('email')}")
    return {"user": user}


@app.post("/add-to-calendar")
async def add_to_calendar(request: Request, event_data: AddEventRequest):
    token = request.session.get('token')
    if not token:
        return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        dt_str = f"{event_data.date_str} {event_data.time_str}"
        start_dt = dateparser.parse(dt_str, settings={'PREFER_DATES_FROM': 'future'})
        if not start_dt:
            start_dt = datetime.datetime.now() + datetime.timedelta(hours=1)
        end_dt = start_dt + datetime.timedelta(hours=1)

        google_event = {
            'summary': event_data.event_str,
            'location': event_data.loc_str,
            'description': event_data.description,
            'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
            'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
        }
        resp = await oauth.google.post(
            'https://www.googleapis.com/calendar/v3/calendars/primary/events',
            json=google_event, token=token
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
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")