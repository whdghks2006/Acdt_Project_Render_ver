import os
import spacy
import dateparser
import datetime
import pandas as pd
import pytz
import httpx
import google.generativeai as genai
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


def extract_schedule_info(translated_text, model_mode='speed'):
    """
    Extracts entities using the selected spaCy model (sm or trf).
    """
    if not translated_text or not translated_text.strip():
        return "", "", "", ""

    # [DUAL ENGINE] Select Model based on user choice
    if model_mode == 'accuracy':
        print("🧠 Using Accuracy Mode (TRF Model)")
        nlp = models["nlp_trf"]
    else:
        print("⚡ Using Speed Mode (SM Model)")
        nlp = models["nlp_sm"]

    doc = nlp(translated_text)

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


def ask_gemini_for_missing_info(text, current_data, lang='ko'):
    if not GEMINI_API_KEY: return ""
    lang_instruction = "in Korean" if lang == 'ko' else "in English"
    prompt = f"""
    You are a helpful scheduler assistant.
    User Input: "{text}"
    Extracted Info:
    - Date: {current_data.get('date', 'Missing')}
    - Time: {current_data.get('time', 'Missing')}
    Task: If Date or Time is MISSING, ask a polite 1-sentence question {lang_instruction}. If present, reply "OK".
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        if "OK" in answer: return ""
        return answer
    except:
        return ""


def save_feedback_to_hub(original_text, translated_text, final_data):
    try:
        if not HF_TOKEN: return
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
        api.upload_file(path_or_fileobj=unique_filename, path_in_repo=unique_filename, repo_id=DATASET_REPO_ID,
                        repo_type="dataset")
        print(f"✅ Feedback data saved.")
    except Exception as e:
        print(f"❌ Failed to save feedback: {e}")


# ==============================================================================
# App Lifecycle (Load BOTH models)
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Loading AI Models...")
    try:
        # Load Speed Model (SM)
        print("⚡ Loading Speed Model (en_core_web_sm)...")
        models["nlp_sm"] = spacy.load("en_core_web_sm")

        # Load Accuracy Model (TRF) - This takes time!
        print("🧠 Loading Accuracy Model (en_core_web_trf)...")
        # Try importing direct package name first (common in requirements install)
        try:
            import en_core_web_trf
            models["nlp_trf"] = en_core_web_trf.load()
        except ImportError:
            models["nlp_trf"] = spacy.load("en_core_web_trf")

        print("✅ All Models loaded successfully!")
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
    lang: str = 'en'  # [Changed] Default 'ko' -> 'en'
    model_mode: str = 'speed'


class ExtractResponse(BaseModel):
    original_text: str
    translated_text: str
    date: str
    time: str
    loc: str
    event: str
    ai_message: str = ""


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
    original_text = request.text
    is_korean_input = check_is_korean(original_text)

    if is_korean_input:
        translated_text = translate_korean_to_english(original_text)
    else:
        translated_text = original_text

    # Pass model_mode to extractor
    date_en, time_en, loc_en, event_en = extract_schedule_info(translated_text, request.model_mode)

    ai_message = ""
    if not date_en or not time_en:
        extracted_data = {'date': date_en, 'time': time_en, 'loc': loc_en}
        ai_message = ask_gemini_for_missing_info(original_text, extracted_data, lang=request.lang)

    if is_korean_input:
        date_final = translate_english_to_korean(date_en)
        time_final = translate_english_to_korean(time_en)
        loc_final = translate_english_to_korean(loc_en)
    else:
        date_final = date_en
        time_final = time_en
        loc_final = loc_en

    return ExtractResponse(
        original_text=original_text, translated_text=translated_text,
        date=date_final, time=time_final, loc=loc_final, event=event_en,
        ai_message=ai_message
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
        kst = pytz.timezone('Asia/Seoul')
        now_kst = datetime.datetime.now(kst)
        settings = {'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_kst.replace(tzinfo=None),
                    'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul', 'RETURN_AS_TIMEZONE_AWARE': True}

        def sanitize_time(text):
            if not text: return ""
            t = text.replace("밤", "오후").replace("저녁", "오후")
            t = t.replace("아침", "오전").replace("새벽", "오전")
            return t

        clean_time_str = sanitize_time(event_data.time_str)
        clean_original_text = sanitize_time(event_data.original_text)

        try:
            date_for_parsing = GoogleTranslator(source='auto', target='en').translate(
                event_data.date_str) if event_data.date_str else ""
            time_for_parsing = GoogleTranslator(source='auto', target='en').translate(
                clean_time_str) if clean_time_str else ""
        except:
            date_for_parsing = event_data.date_str
            time_for_parsing = clean_time_str

        dt_str = f"{date_for_parsing} {time_for_parsing}".strip()
        start_dt = None
        if dt_str:
            start_dt = dateparser.parse(dt_str, settings=settings, languages=['en', 'ko'])

        if not start_dt and clean_original_text:
            try:
                trans_full = GoogleTranslator(source='auto', target='en').translate(clean_original_text)
                start_dt = dateparser.parse(trans_full, settings=settings, languages=['en'])
            except:
                start_dt = dateparser.parse(clean_original_text, settings=settings, languages=['ko'])

        if not start_dt:
            start_dt = now_kst + datetime.timedelta(hours=1)
            start_dt = start_dt.replace(minute=0, second=0, microsecond=0)

        end_dt = start_dt + datetime.timedelta(hours=1)

        google_event = {
            'summary': event_data.event_str,
            'location': event_data.loc_str,
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