import os
import spacy
import dateparser
import datetime
import pandas as pd
import pytz
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

models = {}


# ==============================================================================
# AI Functions
# ==============================================================================
def check_is_korean(text):
    """
    Helper function to detect if text contains Korean characters.
    """
    return any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text)


def translate_korean_to_english(text):
    """
    [UPGRADE] Uses Google Translate (via deep_translator).
    """
    try:
        if check_is_korean(text):
            return GoogleTranslator(source='auto', target='en').translate(text)
        else:
            return text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text


def translate_english_to_korean(text):
    """
    Converts English text back to Korean.
    """
    if not text or not text.strip():
        return ""
    try:
        return GoogleTranslator(source='en', target='ko').translate(text)
    except Exception as e:
        print(f"En->Ko Translation Error: {e}")
        return text


def extract_schedule_info(translated_text):
    """
    Extracts entities using spaCy NER (English).
    """
    if not translated_text or not translated_text.strip():
        return "Please enter text.", "", "", ""

    doc = models["nlp"](translated_text)

    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    times = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
    locs = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    events = [ent.text for ent in doc.ents if ent.label_ == "EVENT"]

    # Return empty string if not found
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


def save_feedback_to_hub(original_text, translated_text, final_data):
    """
    Saves user corrections to Hugging Face Dataset.
    """
    try:
        if not HF_TOKEN:
            return

        new_row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "original_text": original_text,
            "translated_text": translated_text,
            "final_date": final_data.date_str,
            "final_time": final_data.time_str,
            "final_loc": final_data.loc_str,
            "final_event": final_data.event_str
        }
        df = pd.DataFrame([new_row])

        unique_filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(unique_filename, index=False)

        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=unique_filename,
            path_in_repo=unique_filename,
            repo_id=DATASET_REPO_ID,
            repo_type="dataset"
        )
        print(f"✅ Feedback data saved.")

    except Exception as e:
        print(f"❌ Failed to save feedback: {e}")


# ==============================================================================
# App Lifecycle
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
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
    yield
    models.clear()
    print("✅ Models cleared.")


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
    """
    Step 1: Translate -> Extract -> (Optional) Back Translate
    """
    original_text = request.text

    # 1. Check Language
    is_korean_input = check_is_korean(original_text)

    # 2. Translate Ko -> En (if needed)
    if is_korean_input:
        translated_text = translate_korean_to_english(original_text)
    else:
        translated_text = original_text  # English stays English

    # 3. Extract Entities (NER works on English text)
    date_en, time_en, loc_en, event_en = extract_schedule_info(translated_text)

    # 4. Determine Output Language
    if is_korean_input:
        # If input was Korean, show results in Korean (Back Translation)
        date_final = translate_english_to_korean(date_en)
        time_final = translate_english_to_korean(time_en)
        loc_final = translate_english_to_korean(loc_en)
    else:
        # If input was English, show results in English
        date_final = date_en
        time_final = time_en
        loc_final = loc_en

    return ExtractResponse(
        original_text=original_text,
        translated_text=translated_text,
        date=date_final,
        time=time_final,
        loc=loc_final,
        event=event_en
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
        # [DEBUG] Check what the user actually sent
        print(f"📥 User Input - Date: '{event_data.date_str}', Time: '{event_data.time_str}'")

        # 1. Define KST Timezone
        kst = pytz.timezone('Asia/Seoul')
        now_kst = datetime.datetime.now(kst)

        # 2. Settings
        settings = {
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': now_kst.replace(tzinfo=None),
            'TIMEZONE': 'Asia/Seoul',
            'TO_TIMEZONE': 'Asia/Seoul',
            'RETURN_AS_TIMEZONE_AWARE': True
        }

        # 3. [PRIORITY 1] Parse User Edited Boxes First!
        # 사용자가 박스에 적은 내용을 최우선으로 해석합니다.
        dt_str = f"{event_data.date_str} {event_data.time_str}".strip()
        start_dt = None

        if dt_str:
            # Try parsing user input explicitly as Korean/English
            start_dt = dateparser.parse(dt_str, settings=settings, languages=['ko', 'en'])
            if start_dt:
                print(f"✅ Parsed from User Input: {start_dt}")

        # 4. [PRIORITY 2] Smart Fallback (Only if User Input failed or was empty)
        # 사용자가 시간을 안 적었거나, AI가 놓쳤을 때만 원본을 봅니다.
        if not start_dt and event_data.original_text:
            print("⚠️ User input parsing failed/empty. Trying original text...")
            start_dt = dateparser.parse(event_data.original_text, settings=settings, languages=['ko'])
            if start_dt:
                print(f"✅ Recovered from Original Text: {start_dt}")

        # 5. [PRIORITY 3] Final Safety Net (Next Hour)
        # 도저히 날짜를 알 수 없을 때만 현재 시간을 씁니다.
        if not start_dt:
            print("❌ All parsing failed. Defaulting to next hour.")
            start_dt = now_kst + datetime.timedelta(hours=1)
            start_dt = start_dt.replace(minute=0, second=0, microsecond=0)

        end_dt = start_dt + datetime.timedelta(hours=1)

        # 6. Use User Edited Title & Location directly
        # 제목과 장소는 사용자가 수정한 그대로(event_str, loc_str) 넣습니다.
        google_event = {
            'summary': event_data.event_str,
            'location': event_data.loc_str,
            'description': event_data.description,
            'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
            'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
        }

        access_token = token_data['access_token']
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        resp = await oauth.google.post(
            'https://www.googleapis.com/calendar/v3/calendars/primary/events',
            json=google_event, headers=headers
        )

        if resp.status_code != 200:
            if resp.status_code == 401:
                return JSONResponse(status_code=401, content={"error": "Token expired. Login again."})
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
    if key != "1234":
        return HTMLResponse("<h1>🚫 Access Denied</h1>", status_code=403)

    try:
        if not HF_TOKEN:
            return HTMLResponse("<h1>⚠️ HF_TOKEN not set.</h1>")

        api = HfApi(token=HF_TOKEN)
        try:
            files = api.list_repo_files(repo_id=DATASET_REPO_ID, repo_type="dataset")
        except Exception as e:
            return HTMLResponse(f"<h1>❌ Failed to list files.</h1><pre>{str(e)}</pre>")

        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            return HTMLResponse("<h1>📭 No feedback data found yet.</h1>")

        dfs = []
        for file in csv_files:
            try:
                local_filename = hf_hub_download(repo_id=DATASET_REPO_ID, filename=file, repo_type="dataset",
                                                 token=HF_TOKEN)
                df = pd.read_csv(local_filename)
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return HTMLResponse("<h1>❌ Error loading CSV files.</h1>")

        final_df = pd.concat(dfs, ignore_index=True)
        if 'timestamp' in final_df.columns:
            final_df = final_df.sort_values(by='timestamp', ascending=False)

        table_html = final_df.to_html(classes="table table-striped", index=False)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Admin Dashboard</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"><style>body {{ padding: 20px; }}</style></head>
        <body><div class="container"><h1>📊 Feedback Data Log</h1><p>Total Records: {len(final_df)}</p><div class="table-responsive">{table_html}</div></div></body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        return HTMLResponse(f"<h1>❌ System Error: {str(e)}</h1>")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")