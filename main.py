import os
import spacy
import dateparser
import datetime
import pandas as pd
import pytz
import httpx
import google.generativeai as genai
import json
import re
from deep_translator import GoogleTranslator
from huggingface_hub import HfApi, hf_hub_download

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth

# ==============================================================================
# Configuration
# ==============================================================================
NER_MODEL_DIR = "my_ner_model"
DATASET_REPO_ID = "snowmang/scheduler-feedback-data"

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", "random_secret_string")
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

models = {}

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ==============================================================================
# AI Functions
# ==============================================================================
def check_is_korean(text):
    return any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text)


def translate_korean_to_english(text):
    try:
        if check_is_korean(text):
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text


def run_intelligent_analysis(text, lang='en'):
    """
    Gemini extracts Start/End Date AND Start/End Time separately.
    """
    if not GEMINI_API_KEY: return None

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    lang_instruction = "in Korean" if lang == 'ko' else "in English"

    prompt = f"""
    You are a smart scheduler assistant. Today is {today}.
    User Input: "{text}"

    Your Task:
    1. Extract schedule details.
       - Summary: Concise title (e.g. "Dinner at Hongdae").
       - Date Range: If "20th to 23rd", start=20, end=23. If single day, start=end.
       - Time Range: If "2pm to 4pm", start_time="14:00", end_time="16:00". 
         If only "2pm", start_time="14:00", end_time="".

    2. Generate a follow-up question ONLY if critical info is missing.
       - If Date is missing, ask for Date.
       - If Time is missing BUT 'is_allday' seems false, ask for Time.
       - Question should be polite and {lang_instruction}.
       - If info is complete, set question to "".

    Output JSON format:
    {{
      "summary": "Event Title",
      "start_date": "YYYY-MM-DD" or "",
      "end_date": "YYYY-MM-DD" or "",
      "start_time": "HH:MM" or "",
      "end_time": "HH:MM" or "",
      "location": "Location" or "",
      "is_allday": true/false,
      "question": "Follow-up question or empty string"
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        clean_text = re.sub(r'```json|```', '', response.text).strip()
        data = json.loads(clean_text)
        return data
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None


def save_feedback_to_hub(original_text, translated_text, final_data):
    try:
        if not HF_TOKEN: return
        new_row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "original_text": original_text,
            "final_summary": final_data.summary,
            "final_start": final_data.start_date,
            "final_end": final_data.end_date,
            "final_loc": final_data.location
        }
        df = pd.DataFrame([new_row])
        unique_filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(unique_filename, index=False)

        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=unique_filename, path_in_repo=unique_filename, repo_id=DATASET_REPO_ID,
                        repo_type="dataset")
        print(f"✅ Feedback saved.")
    except Exception as e:
        print(f"❌ Save Error: {e}")


# ==============================================================================
# App Lifecycle
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ App Started")
    yield
    print("✅ App Shutdown")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=True,
    same_site='none',
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


# --- Models ---
class ExtractRequest(BaseModel):
    text: str
    lang: str = 'en'


class ExtractResponse(BaseModel):
    original_text: str
    translated_text: str
    summary: str
    start_date: str
    end_date: str
    start_time: str  # [New]
    end_time: str  # [New]
    location: str
    is_allday: bool
    ai_message: str = ""
    used_model: str = ""


class AddEventRequest(BaseModel):
    summary: str
    start_date: str
    end_date: str
    start_time: str  # [New]
    end_time: str  # [New]
    location: str
    description: str
    is_allday: bool
    original_text: str
    translated_text: str
    consent: bool = False


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
    is_korean = check_is_korean(original_text)
    translated_text = translate_korean_to_english(original_text) if is_korean else original_text

    data = run_intelligent_analysis(original_text, lang=request.lang)

    if data:
        return ExtractResponse(
            original_text=original_text,
            translated_text=translated_text,
            summary=data.get("summary", ""),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            start_time=data.get("start_time", ""),  # [New]
            end_time=data.get("end_time", ""),  # [New]
            location=data.get("location", ""),
            is_allday=data.get("is_allday", False),
            ai_message=data.get("question", ""),
            used_model="Gemini 2.0 Flash"
        )
    else:
        return ExtractResponse(
            original_text=original_text, translated_text=translated_text,
            summary="", start_date="", end_date="", start_time="", end_time="",
            location="", is_allday=False,
            ai_message="Failed to analyze.", used_model="Error"
        )


@app.get('/login')
async def login(request: Request):
    fixed_redirect_uri = "https://snowmang-ai-scheduler-g14.hf.space/auth/callback"
    return await oauth.google.authorize_redirect(request, fixed_redirect_uri)


@app.get('/auth/callback')
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        request.session['user'] = {'name': user_info.get('name'), 'email': user_info.get('email')}
        request.session['token'] = {'access_token': token.get('access_token'), 'token_type': token.get('token_type')}
        return RedirectResponse(url='/', status_code=303)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Login failed: {str(e)}"})


@app.get('/logout')
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url='/')


@app.get('/user-info')
async def get_user_info(request: Request):
    return {"user": request.session.get('user')}


@app.post("/add-to-calendar")
async def add_to_calendar(request: Request, event_data: AddEventRequest):
    token_data = request.session.get('token')
    if not token_data or 'access_token' not in token_data:
        return JSONResponse(status_code=401, content={"error": "Login required"})

    try:
        # [FIX] Advanced Date Logic for Full Days & Ranges
        s_date_obj = dateparser.parse(event_data.start_date)
        e_date_obj = dateparser.parse(event_data.end_date)
        if not s_date_obj: s_date_obj = datetime.datetime.now()
        if not e_date_obj: e_date_obj = s_date_obj

        if event_data.is_allday:
            # [All Day Logic]
            # Google Calendar treats end date as exclusive.
            # If User says "20th to 23rd", they mean 23rd is included.
            # So we must send "20th" as start and "24th" as end to Google.
            s_str = s_date_obj.strftime("%Y-%m-%d")

            # Add 1 day to end date for Google's exclusive logic
            e_date_exclusive = e_date_obj + datetime.timedelta(days=1)
            e_str = e_date_exclusive.strftime("%Y-%m-%d")

            google_event = {
                'summary': event_data.summary,
                'location': event_data.location,
                'description': event_data.description,
                'start': {'date': s_str},
                'end': {'date': e_str},
            }
        else:
            # [Timed Logic]
            kst = pytz.timezone('Asia/Seoul')

            # Parse Start Time
            start_full_str = f"{event_data.start_date} {event_data.start_time}"
            start_dt = dateparser.parse(start_full_str, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
                                                                  'RETURN_AS_TIMEZONE_AWARE': True})

            # Parse End Time (if exists) or Default +1 Hour
            if event_data.end_time:
                end_full_str = f"{event_data.end_date} {event_data.end_time}"
                end_dt = dateparser.parse(end_full_str, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
                                                                  'RETURN_AS_TIMEZONE_AWARE': True})
            else:
                if not start_dt: start_dt = datetime.datetime.now(kst)
                end_dt = start_dt + datetime.timedelta(hours=1)

            google_event = {
                'summary': event_data.summary,
                'location': event_data.location,
                'description': event_data.description,
                'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
                'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
            }

        access_token = token_data['access_token']
        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                'https://www.googleapis.com/calendar/v3/calendars/primary/events',
                json=google_event, headers=headers
            )

        if resp.status_code != 200:
            if resp.status_code == 401: return JSONResponse(status_code=401, content={"error": "Token expired."})
            print(f"Google API Error: {resp.text}")
            resp.raise_for_status()

        result = resp.json()

        if event_data.consent:
            save_feedback_to_hub(event_data.original_text, event_data.translated_text, event_data)
            saved_msg = "✅ Data saved."
        else:
            saved_msg = "ℹ️ Data NOT saved."

        return {"message": "Success", "link": result.get('htmlLink'), "saved_msg": saved_msg}

    except Exception as e:
        print(f"Calendar Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    key = request.query_params.get("key")
    if key != "1234": return HTMLResponse("<h1>🚫 Access Denied</h1>", status_code=403)
    # ... Admin code ...
    return HTMLResponse("<h1>Admin Dashboard</h1>")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")