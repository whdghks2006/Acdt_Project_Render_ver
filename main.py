import os
import spacy
import dateparser
import datetime
import pandas as pd
import pytz
import httpx
import google.generativeai as genai
import json
from urllib.parse import quote_plus
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

# Secrets
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", "random_secret_string")
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

models = {}

# Initialize LLM Engine
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ==============================================================================
# AI Functions (Logic Layer)
# ==============================================================================
def check_is_korean(text):
    return any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text)


def translate_korean_to_english(text):
    try:
        if check_is_korean(text):
            return GoogleTranslator(source='auto', target='en').translate(text)
        else:
            return text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text


def translate_english_to_korean(text):
    if not text or not text.strip(): return ""
    try:
        return GoogleTranslator(source='en', target='ko').translate(text)
    except Exception as e:
        print(f"En->Ko Translation Error: {e}")
        return text


def run_ner_extraction(text, nlp_model):
    """
    Core NER logic using our spaCy model.
    """
    doc = nlp_model(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    times = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
    locs = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    events = [ent.text for ent in doc.ents if ent.label_ == "EVENT"]

    date_str = ", ".join(dates) if dates else ""
    time_str = ", ".join(times) if times else ""
    loc_str = ", ".join(locs) if locs else ""

    if events:
        event_str = ", ".join(events)
    elif locs:
        event_str = f"Meeting at {loc_str}"
    else:
        event_str = "New Schedule"

    return date_str, time_str, loc_str, event_str


def run_intelligent_gap_filling(text, lang='ko'):
    """
    Unified Context Analysis & Gap Filling Agent
    """
    if not GEMINI_API_KEY: return None, None

    lang_instruction = "in Korean" if lang == 'ko' else "in English"
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Prompt
    prompt = f"""
    You are an intelligent scheduling assistant. Today is {today}.
    User Input: "{text}"

    Your Goal: 
    1. Parse the input into structured data (Date, Time, Location, Event).
       - Handle ranges (e.g., "Nov 20-23").
       - If single date, start=end.
       - If no time specified, is_allday=true.

    2. **Create a Professional Event Title (summary)**:
       - Summarize the *action* or *topic* concisely (2-5 words).
       - If input is Korean, title MUST be Korean.

    3. Quality Check:
       - If 'Date' or 'Time' is missing, formulate a polite follow-up question {lang_instruction}.
       - If complete, set 'question' to empty string "".

    Output JSON ONLY:
    {{
      "summary": "Event Title",
      "start_date": "YYYY-MM-DD" or "",
      "end_date": "YYYY-MM-DD" or "",
      "time": "HH:MM" or "",
      "location": "Location" or "",
      "is_allday": true/false,
      "question": "Follow-up question or empty"
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_text)

        print(f"🤖 Agent Analysis Result: {data}")
        return data, data.get("question", "")
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        return None, ""


def save_feedback_to_hub(original_text, translated_text, final_data):
    try:
        if not HF_TOKEN: return
        new_row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "original_text": original_text,
            "final_start_date": final_data.start_date,
            "final_end_date": final_data.end_date,
            "final_time": final_data.time,
            "final_loc": final_data.location,
            "final_event": final_data.summary
        }
        df = pd.DataFrame([new_row])
        unique_filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(unique_filename, index=False)
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=unique_filename, path_in_repo=unique_filename, repo_id=DATASET_REPO_ID,
                        repo_type="dataset")
        print(f"✅ Feedback data saved.")
    except Exception as e:
        print(f"❌ Failed to save feedback: {e}")


# ==============================================================================
# App Lifecycle
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 System Startup: Loading Neural Modules...")
    try:
        print("⚡ Initializing Fast-Inference Module (SM)...")
        models["nlp_sm"] = spacy.load("en_core_web_sm")
        print("✅ Core modules ready.")
    except Exception as e:
        print(f"❌ Init Error: {e}")
    yield
    models.clear()
    print("✅ System Shutdown.")


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


class ExtractRequest(BaseModel):
    text: str
    lang: str = 'en'


# [FIXED] Updated Model to match new logic
class ExtractResponse(BaseModel):
    original_text: str
    summary: str  # Changed from 'event' to 'summary'
    start_date: str  # Changed from 'date' to 'start_date'
    end_date: str  # Added 'end_date'
    time: str
    location: str  # Changed from 'loc' to 'location'
    is_allday: bool  # Added 'is_allday'
    ai_message: str = ""
    used_model: str = ""


class AddEventRequest(BaseModel):
    summary: str
    start_date: str
    end_date: str
    time: str
    location: str
    description: str
    is_allday: bool
    original_text: str
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
    """
    Pipeline: Fast NER -> Intelligent Gap Filling (Agent)
    """
    original_text = request.text
    is_korean_input = check_is_korean(original_text)

    # 1. Pre-processing
    if is_korean_input:
        translated_text = translate_korean_to_english(original_text)
    else:
        translated_text = original_text

    date, time, loc, event = run_ner_extraction(translated_text, models["nlp_sm"])
    used_model = "Fast-Inference (NER)"

    # 2. Intelligent Analysis
    agent_data = None
    ai_message = ""

    if not date or not time:
        print("⚠️ Insufficient data. Activating Dialogue Engine...")
        agent_data, ai_message = run_intelligent_gap_filling(original_text, lang=request.lang)
        used_model = "Dialogue Engine (Generative)"

    # 3. Response Construction
    if agent_data:
        summary_final = agent_data.get("summary", event)
        s_date_final = agent_data.get("start_date", "")
        e_date_final = agent_data.get("end_date", "")
        time_final = agent_data.get("time", "")
        loc_final = agent_data.get("location", loc)
        is_allday = agent_data.get("is_allday", False)
    else:
        # Fallback: Convert spaCy single date to start/end format
        summary_final = original_text if is_korean_input else event
        s_date_final = date
        e_date_final = date  # Default to single day
        time_final = time
        loc_final = loc
        is_allday = False

        if is_korean_input:
            s_date_final = translate_english_to_korean(s_date_final)
            e_date_final = s_date_final
            time_final = translate_english_to_korean(time_final)
            loc_final = translate_english_to_korean(loc_final)

    return ExtractResponse(
        original_text=original_text,
        summary=summary_final,
        start_date=s_date_final,
        end_date=e_date_final,
        time=time_final,
        location=loc_final,
        is_allday=is_allday,
        ai_message=ai_message,
        used_model=used_model
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
        if event_data.is_allday:
            # Parse YYYY-MM-DD strings to date objects
            s_date = dateparser.parse(event_data.start_date).date()
            e_date = dateparser.parse(event_data.end_date).date()

            # Google Calendar Logic: End date is exclusive
            # If user says "20th to 23rd", they mean inclusive. Google needs 24th as end.
            if s_date == e_date:
                e_date = s_date + datetime.timedelta(days=1)  # 1 day event
            else:
                e_date = e_date + datetime.timedelta(days=1)  # Add 1 day to include the end date

            google_event = {
                'summary': event_data.summary,
                'location': event_data.location,
                'description': event_data.description,
                'start': {'date': s_date.isoformat()},
                'end': {'date': e_date.isoformat()},
            }
        else:
            kst = pytz.timezone('Asia/Seoul')
            start_str = f"{event_data.start_date} {event_data.time}"
            start_dt = dateparser.parse(start_str)
            if not start_dt: raise ValueError("Invalid Date/Time format")
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
            save_feedback_to_hub(event_data.original_text, "", event_data)
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
    # (Admin logic - keep as is or copy from previous)
    try:
        if not HF_TOKEN: return HTMLResponse("<h1>⚠️ HF_TOKEN not set.</h1>")
        api = HfApi(token=HF_TOKEN)
        try:
            files = api.list_repo_files(repo_id=DATASET_REPO_ID, repo_type="dataset")
        except Exception as e:
            return HTMLResponse(f"<h1>❌ Failed to list files.</h1><pre>{str(e)}</pre>")

        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files: return HTMLResponse("<h1>📭 No data found.</h1>")

        dfs = []
        for file in csv_files:
            try:
                local_filename = hf_hub_download(repo_id=DATASET_REPO_ID, filename=file, repo_type="dataset",
                                                 token=HF_TOKEN)
                df = pd.read_csv(local_filename)
                dfs.append(df)
            except Exception:
                continue

        if not dfs: return HTMLResponse("<h1>❌ Error loading CSV.</h1>")
        final_df = pd.concat(dfs, ignore_index=True)
        if 'timestamp' in final_df.columns: final_df = final_df.sort_values(by='timestamp', ascending=False)
        table_html = final_df.to_html(classes="table table-striped", index=False)
        return HTMLResponse(
            f"<html><head><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'></head><body><div class='container mt-4'><h1>📊 Feedback Log</h1><p>Total: {len(final_df)}</p>{table_html}</div></body></html>")
    except Exception as e:
        return HTMLResponse(f"<h1>❌ Error: {str(e)}</h1>")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")