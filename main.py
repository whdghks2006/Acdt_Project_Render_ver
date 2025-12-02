import os
from dotenv import load_dotenv

load_dotenv()

import spacy
import dateparser
import datetime
import pandas as pd
import pytz
import httpx
import google.generativeai as genai
import json
import re
import io
import PIL.Image  # 이미지 처리를 위한 라이브러리

from deep_translator import GoogleTranslator
from huggingface_hub import HfApi, hf_hub_download

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth

# ==============================================================================
# Configuration
# ==============================================================================
NER_MODEL_NAME = "en_core_web_md"  # 문맥 이해도가 높은 Medium 모델 사용
DATASET_REPO_ID = "snowmang/scheduler-feedback-data"

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", "random_secret_string")
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

models = {}

print(f"🔑 GEMINI_API_KEY Loaded: {bool(GEMINI_API_KEY)}")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ==============================================================================
# Helper Functions (Language & Translation)
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


def translate_english_to_korean(text):
    if not text or not text.strip(): return ""
    try:
        return GoogleTranslator(source='en', target='ko').translate(text)
    except:
        return text


# [NEW] Smart Model Selector (Debugging Logic)
# 2.5 버전을 먼저 시도하고, 실패하면(404 등) 1.5로 자동 전환하여 서버 다운 방지
def get_gemini_content(prompt, image=None, target_model="gemini-2.5-flash"):
    try:
        model = genai.GenerativeModel(target_model)
        if image:
            return model.generate_content([prompt, image])
        return model.generate_content(prompt)
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ {target_model} failed: {error_msg}")

        # 429 Quota Error는 모델 문제가 아니므로 즉시 에러 반환 (재시도 X)
        if "429" in error_msg or "Resource exhausted" in error_msg:
            raise HTTPException(status_code=429, detail="Google AI Quota Exceeded. Please try again in 1 min.")

        # 그 외 에러(모델 없음 등)는 1.5로 폴백
        print(f"🔄 Falling back to gemini-1.5-flash...")
        fallback_model = genai.GenerativeModel("gemini-1.5-flash")
        if image:
            return fallback_model.generate_content([prompt, image])
        return fallback_model.generate_content(prompt)


# ==============================================================================
# AI Logic 1: Text Analysis (spaCy + Gemini Hybrid)
# ==============================================================================
def run_ner_extraction(text, nlp_model):
    """spaCy를 이용한 빠른 1차 추출"""
    if not text: return "", "", "", ""
    doc = nlp_model(text)

    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    times = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
    locs = [ent.text for ent in doc.ents if ent.label_ == "LOC" or ent.label_ == "GPE"]
    events = [ent.text for ent in doc.ents if ent.label_ == "EVENT"]

    date_str = ", ".join(dates) if dates else ""
    time_str = ", ".join(times) if times else ""
    loc_str = ", ".join(locs) if locs else ""

    # 이벤트 제목 추론 (Heuristic)
    if events:
        event_str = ", ".join(events)
    elif locs:
        event_str = f"Meeting at {loc_str}"
    else:
        event_str = "New Schedule"

    return date_str, time_str, loc_str, event_str


def extract_info_with_gemini_json(text):
    """Gemini를 이용한 정밀 2차 추출"""
    if not GEMINI_API_KEY: return None
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    You are a smart scheduler assistant. Today is {today}.
    Extract schedule details from: "{text}"
    (Note: If this looks like a chat log, ignore message timestamps and focus on the conversation content.)

    Rules:
    1. Handle date ranges (start_date, end_date). If single day, start=end.
    2. Summarize the event title concisely.
    3. If no time is mentioned, set is_allday to true.
    4. Separate start_time and end_time if a range is given (e.g. 2pm-4pm).

    Return JSON ONLY: 
    {{ "summary": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM", "location": "...", "is_allday": boolean }}
    """
    try:
        # [Updated] Try 2.5 -> Fallback 1.5
        response = get_gemini_content(prompt, target_model='gemini-2.5-flash')
        clean = re.sub(r'```json|```', '', response.text).strip()
        return json.loads(clean)
    except Exception as e:
        print(f"Gemini Text Error: {e}")
        return None


def ask_gemini_for_missing_info(text, current_data, lang='en'):
    """부족한 정보에 대한 역질문 생성"""
    if not GEMINI_API_KEY: return ""
    lang_instruction = "in Korean" if lang == 'ko' else "in English"

    prompt = f"""
    User Input: "{text}"
    Current Info: {current_data}

    Task:
    Check if 'Date' or 'Time' (or 'is_allday') is missing.
    If missing, ask a polite question {lang_instruction} to get that info.
    If complete, return "OK".
    """
    try:
        # [Updated] Try 2.5 -> Fallback 1.5
        response = get_gemini_content(prompt, target_model='gemini-2.5-flash')
        ans = response.text.strip()
        return "" if "OK" in ans else ans
    except:
        return ""


# ==============================================================================
# AI Logic 2: Vision Analysis (Image/Screenshot)
# ==============================================================================

def run_vision_transcription(image_bytes):
    """이미지에서 텍스트만 추출 (OCR)"""
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY is missing.")
        return ""
    
    try:
        image = PIL.Image.open(io.BytesIO(image_bytes))
        prompt = "Transcribe all text from this image exactly as it appears. Do not summarize."
        
        # Try 2.5 -> Fallback 1.5
        response = get_gemini_content(prompt, image=image, target_model='gemini-2.5-flash')
        if not response or not response.text:
            print("❌ Gemini returned empty response.")
            return ""
        return response.text.strip()
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Vision Transcription Error: {e}")
        return ""



def run_vision_analysis(image_bytes):
    """이미지를 분석하여 일정 정보 추출"""
    if not GEMINI_API_KEY: return {"error": "GEMINI_API_KEY not set"}
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        image = PIL.Image.open(io.BytesIO(image_bytes))
        prompt = f"""
        You are a smart scheduler assistant. Today is {today}.
        Analyze this image (screenshot of chat conversation or document).

        Your Task:
        1. Read all text in the image (Supports Korean & English).
        2. Identify schedule details (Summary, Date, Time, Location).
        3. If the text is in Korean, the 'summary' MUST be in Korean.
        4. Detect start and end dates accurately.

        Return JSON ONLY:
        {{
          "summary": "...",
          "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD",
          "start_time": "HH:MM", "end_time": "HH:MM",
          "location": "...", "is_allday": boolean,
          "question": "Follow-up question if info is missing"
        }}
        """
        # [Updated] Try 2.5 -> Fallback 1.5 (With Error Reporting)
        response = get_gemini_content(prompt, image=image, target_model='gemini-2.5-flash')
        clean = re.sub(r'```json|```', '', response.text).strip()
        return json.loads(clean)

    except HTTPException as he:
        # 429 에러는 그대로 전달
        return {"error": he.detail, "error_code": he.status_code}
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Vision Analysis Error: {error_msg}")
        return {"error": error_msg}


# ==============================================================================
# Data Handling & Lifecycle
# ==============================================================================
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ App Started")
    # Load spaCy model on startup
    try:
        if not spacy.util.is_package(NER_MODEL_NAME):
            print(f"⚠️ Downloading {NER_MODEL_NAME}...")
            spacy.cli.download(NER_MODEL_NAME)
        models["nlp_sm"] = spacy.load(NER_MODEL_NAME)
        print(f"✅ spaCy model '{NER_MODEL_NAME}' loaded.")
    except Exception as e:
        print(f"❌ Failed to load spaCy: {e}")
    yield
    print("✅ App Shutdown")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=False,
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


# --- Pydantic Models ---
class ExtractRequest(BaseModel):
    text: str
    lang: str = 'en'
    mode: str = "full"  # 'fast' or 'full'


class ExtractResponse(BaseModel):
    original_text: str
    translated_text: str
    summary: str
    start_date: str
    end_date: str
    start_time: str
    end_time: str
    location: str
    is_allday: bool
    ai_message: str = ""
    used_model: str = ""
    spacy_log: str = ""


class AddEventRequest(BaseModel):
    summary: str
    start_date: str
    end_date: str
    start_time: str
    end_time: str
    location: str
    description: str
    is_allday: bool
    original_text: str
    translated_text: str
    consent: bool = False


class UpdateEventRequest(BaseModel):
    summary: str
    location: str
    description: str
    start_date: str | None = None
    end_date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    is_allday: bool = False


# ==============================================================================
# Endpoints
# ==============================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


# --- 1. Text Analysis (Hybrid: Fast -> Smart) ---

def process_text_schedule(text: str, mode: str = "full", lang: str = "en", is_ocr: bool = False):
    """Shared logic for text analysis (Text -> spaCy -> Gemini)"""
    original_text = text
    is_korean_input = check_is_korean(original_text)

    # 1. Translate & spaCy (Fast)
    translated_text = translate_korean_to_english(original_text) if is_korean_input else original_text

    date_str, time_str, loc_str, event_str = "", "", "", ""
    if "nlp_sm" in models:
        date_str, time_str, loc_str, event_str = run_ner_extraction(translated_text, models["nlp_sm"])

    spacy_debug_str = f"Date=[{date_str}] Time=[{time_str}] Loc=[{loc_str}]"

    # Set Initial Values
    summary_val = original_text if is_korean_input else event_str
    start_date_val = date_str
    end_date_val = date_str
    start_time_val = time_str
    end_time_val = ""
    loc_val = loc_str
    is_allday_val = False
    used_model = "Fast-Inference (spaCy)"
    ai_message = ""

    # [Branch] If fast mode, return immediately
    if mode == "fast":
        if is_korean_input: loc_val = translate_english_to_korean(loc_val)
        return ExtractResponse(
            original_text=original_text, translated_text=translated_text,
            summary=summary_val, start_date=start_date_val, end_date=end_date_val,
            start_time=start_time_val, end_time=end_time_val,
            location=loc_val, is_allday=is_allday_val,
            ai_message="", used_model="⚡ Fast (spaCy)",
            spacy_log=spacy_debug_str
        )

    # 2. Gemini Extraction (Smart Fallback)
    # If spaCy missed something OR if we are in 'full' mode and want to be sure
    if is_ocr or not date_str or not time_str or " to " in translated_text:
        print("⚠️ spaCy incomplete. Calling Gemini...")
        gemini_data = extract_info_with_gemini_json(original_text)

        if gemini_data:
            summary_val = gemini_data.get("summary") or summary_val
            start_date_val = gemini_data.get("start_date") or ""
            end_date_val = gemini_data.get("end_date") or ""

            g_start = gemini_data.get("start_time") or gemini_data.get("time")
            start_time_val = g_start or ""
            end_time_val = gemini_data.get("end_time") or ""

            loc_val = gemini_data.get("location") or loc_val
            is_allday_val = gemini_data.get("is_allday") or False
            used_model = "✨ Smart (Gemini 2.5)"

    # 3. Localization
    if is_korean_input and used_model.startswith("Fast"):
        loc_val = translate_english_to_korean(loc_val)

    # 4. Interactive Question
    if not start_date_val or (not start_time_val and not is_allday_val):
        current_data = {'date': start_date_val, 'time': start_time_val, 'loc': loc_val}
        ai_message = ask_gemini_for_missing_info(original_text, current_data, lang=lang)

    return ExtractResponse(
        original_text=original_text, translated_text=translated_text,
        summary=summary_val, start_date=start_date_val, end_date=end_date_val,
        start_time=start_time_val, end_time=end_time_val,
        location=loc_val, is_allday=is_allday_val,
        ai_message=ai_message, used_model=used_model,
        spacy_log=spacy_debug_str
    )


@app.post("/extract", response_model=ExtractResponse)
async def api_extract_schedule(request: ExtractRequest):
    return process_text_schedule(request.text, request.mode, request.lang)


# --- 2. Image Analysis (Vision) ---
@app.post("/extract-image", response_model=ExtractResponse)
async def api_extract_image_schedule(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # 1. Transcribe Image to Text
        transcribed_text = run_vision_transcription(contents)
        
        if not transcribed_text:
            return JSONResponse(status_code=500, content={"error": "Failed to read text from image."})
            
        print(f"📷 Image Transcribed: {transcribed_text[:50]}...")

        # 2. Process as Text (spaCy -> Gemini)
        # We force 'full' mode to ensure high quality extraction from OCR text
        result = process_text_schedule(transcribed_text, mode="full", is_ocr=True)
        
        # Update model name to indicate source
        result.used_model = f"Image OCR + {result.used_model}"
        return result

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    token_data = request.session.get('token')
    if not token_data or 'access_token' not in token_data:
        return JSONResponse(status_code=401, content={"error": "Login required"})

    try:
        s_date_obj = dateparser.parse(event_data.start_date)
        e_date_obj = dateparser.parse(event_data.end_date)
        if not s_date_obj: s_date_obj = datetime.datetime.now()
        if not e_date_obj: e_date_obj = s_date_obj

        if event_data.is_allday:
            s_str = s_date_obj.strftime("%Y-%m-%d")
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
            kst = pytz.timezone('Asia/Seoul')
            start_full = f"{event_data.start_date} {event_data.start_time}"
            start_dt = dateparser.parse(start_full, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
                                                              'RETURN_AS_TIMEZONE_AWARE': True})

            if event_data.end_time:
                end_full = f"{event_data.end_date} {event_data.end_time}"
                end_dt = dateparser.parse(end_full, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
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


@app.get("/events")
async def list_events(request: Request):
    token_data = request.session.get('token')
    if not token_data: return JSONResponse(status_code=401, content={"error": "Login required"})

    access_token = token_data['access_token']
    headers = {'Authorization': f'Bearer {access_token}'}
    now = datetime.datetime.utcnow().isoformat() + 'Z'

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get('https://www.googleapis.com/calendar/v3/calendars/primary/events', headers=headers,
                                    params={'timeMin': now, 'maxResults': 100, 'singleEvents': True,
                                            'orderBy': 'startTime'})

        if resp.status_code != 200: return JSONResponse(status_code=resp.status_code,
                                                        content={"error": "Failed to fetch"})

        items = resp.json().get('items', [])
        calendar_events = []
        for event in items:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            is_allday = 'date' in event['start']

            calendar_events.append({
                'id': event['id'],
                'title': event.get('summary', 'No Title'),
                'start': start,
                'end': end,
                'allDay': is_allday,
                'url': event.get('htmlLink'),
                'extendedProps': {'description': event.get('description', ''), 'location': event.get('location', '')}
            })
        return {"events": calendar_events}
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.patch("/events/{event_id}")
async def update_event(request: Request, event_id: str, event_data: UpdateEventRequest):
    token_data = request.session.get('token')
    if not token_data: return JSONResponse(status_code=401, content={"error": "Login required"})

    headers = {'Authorization': f'Bearer {token_data["access_token"]}', 'Content-Type': 'application/json'}

    body = {
        "summary": event_data.summary,
        "location": event_data.location,
        "description": event_data.description
    }

    try:
        if event_data.start_date:
            s_date_obj = dateparser.parse(event_data.start_date)
            e_date_obj = dateparser.parse(event_data.end_date) if event_data.end_date else s_date_obj

            if event_data.is_allday:
                s_str = s_date_obj.strftime("%Y-%m-%d")
                e_date_exclusive = e_date_obj + datetime.timedelta(days=1)
                e_str = e_date_exclusive.strftime("%Y-%m-%d")
                body['start'] = {'date': s_str}
                body['end'] = {'date': e_str}
            else:
                start_full = f"{event_data.start_date} {event_data.start_time}"
                start_dt = dateparser.parse(start_full, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
                                                                  'RETURN_AS_TIMEZONE_AWARE': True})

                end_full = f"{event_data.end_date} {event_data.end_time}"
                end_dt = dateparser.parse(end_full, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
                                                              'RETURN_AS_TIMEZONE_AWARE': True})

                body['start'] = {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Seoul'}
                body['end'] = {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Seoul'}

    except Exception as e:
        print(f"Date Parse Error in Update: {e}")
        pass

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.patch(f'https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}',
                                      json=body, headers=headers)

        if resp.status_code != 200: return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
        return {"message": "Updated successfully"}
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/events/{event_id}")
async def delete_event(request: Request, event_id: str):
    token_data = request.session.get('token')
    if not token_data: return JSONResponse(status_code=401, content={"error": "Login required"})

    headers = {'Authorization': f'Bearer {token_data["access_token"]}'}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(f'https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}',
                                       headers=headers)

        if resp.status_code != 204: return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
        return {"message": "Deleted successfully"}
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
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