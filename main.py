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


def translate_english_to_korean(text):
    if not text or not text.strip(): return ""
    try:
        return GoogleTranslator(source='en', target='ko').translate(text)
    except:
        return text


# [1] Fast NER (spaCy)
def run_ner_extraction(text, nlp_model):
    if not text: return "", "", "", ""
    doc = nlp_model(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    times = [ent.text for ent in doc.ents if ent.label_ == "TIME"]
    locs = [ent.text for ent in doc.ents if ent.label_ == "LOC" or ent.label_ == "GPE"]
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


# [2] Smart Extraction (Gemini)
def extract_info_with_gemini_json(text):
    if not GEMINI_API_KEY: return None
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    You are a smart scheduler assistant. Today is {today}.
    Extract schedule details from: "{text}"
    Rules:
    1. Handle date ranges. If single day, start=end.
    2. Summarize title concisely.
    3. If no time, set is_allday=true.
    Return JSON ONLY: {{ "summary": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "time": "HH:MM", "location": "...", "is_allday": boolean }}
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        clean = re.sub(r'```json|```', '', response.text).strip()
        return json.loads(clean)
    except:
        return None


# [3] Interactive Question (Gemini)
def ask_gemini_for_missing_info(text, current_data, lang='en'):
    if not GEMINI_API_KEY: return ""
    lang_instruction = "in Korean" if lang == 'ko' else "in English"
    prompt = f"""
    User Input: "{text}"
    Current Info: {current_data}
    Task: If Date or Time is missing, ask a polite question {lang_instruction}. If complete, return "OK".
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        ans = response.text.strip()
        return "" if "OK" in ans else ans
    except:
        return ""


def save_feedback_to_hub(original_text, translated_text, final_data):
    try:
        if not HF_TOKEN: return
        new_row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "original_text": original_text,
            "final_summary": final_data.summary,
            "final_start": final_data.start_date,
            "final_end": final_data.end_date,
            "final_time": final_data.time if hasattr(final_data, 'time') else final_data.start_time,
            "final_loc": final_data.location
        }
        df = pd.DataFrame([new_row])
        unique_filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(unique_filename, index=False)
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=unique_filename, path_in_repo=unique_filename, repo_id=DATASET_REPO_ID,
                        repo_type="dataset")
        print("✅ Feedback saved.")
    except Exception as e:
        print(f"❌ Save Error: {e}")


# ==============================================================================
# App Lifecycle
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 System Startup...")
    try:
        print("⚡ Loading spaCy (Fast)...")
        models["nlp"] = spacy.load("en_core_web_sm")
        print("✅ Models loaded!")
    except Exception as e:
        print(f"❌ Load Error: {e}")
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=True, same_site='none', path='/', max_age=3600)
oauth = OAuth()
oauth.register(name='google', client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
               server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
               client_kwargs={'scope': 'openid email profile https://www.googleapis.com/auth/calendar.events'})


# --- Models ---
class ExtractRequest(BaseModel):
    text: str
    lang: str = 'en'


class ExtractResponse(BaseModel):
    original_text: str;
    translated_text: str;
    summary: str;
    start_date: str;
    end_date: str;
    time: str;
    location: str;
    is_allday: bool;
    ai_message: str = "";
    used_model: str = ""


class AddEventRequest(BaseModel):
    summary: str;
    start_date: str;
    end_date: str;
    start_time: str;
    end_time: str;
    location: str;
    description: str;
    is_allday: bool;
    original_text: str;
    translated_text: str;
    consent: bool = False


class UpdateEventRequest(BaseModel):
    summary: str;
    location: str;
    description: str


# ==============================================================================
# Endpoints
# ==============================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index(): return FileResponse('static/index.html')


@app.post("/extract", response_model=ExtractResponse)
async def api_extract_schedule(request: ExtractRequest):
    """ Pipeline: spaCy -> Gemini Extraction -> Gemini Question """
    original_text = request.text
    is_korean_input = check_is_korean(original_text)

    # 1. spaCy (Fast)
    translated_text = translate_korean_to_english(original_text) if is_korean_input else original_text
    date, time, loc, event = run_ner_extraction(translated_text, models["nlp"])

    summary_val = original_text if is_korean_input else event
    start_date_val = date
    end_date_val = date
    time_val = time
    loc_val = loc
    is_allday_val = False
    used_model = "Fast-Inference (spaCy)"

    # 2. Gemini Extraction (Smart)
    if not date or not time or " to " in translated_text:
        print("⚠️ spaCy incomplete. Calling Gemini...")
        gemini_data = extract_info_with_gemini_json(original_text)
        if gemini_data:
            summary_val = gemini_data.get("summary", summary_val)
            start_date_val = gemini_data.get("start_date", "")
            end_date_val = gemini_data.get("end_date", "")
            time_val = gemini_data.get("time", "")
            loc_val = gemini_data.get("location", loc_val)
            is_allday_val = gemini_data.get("is_allday", False)
            used_model = "Smart (Gemini 2.0)"

    # 3. Localization
    if is_korean_input and used_model.startswith("Fast"):
        start_date_val = translate_english_to_korean(start_date_val)
        end_date_val = start_date_val
        time_val = translate_english_to_korean(time_val)
        loc_val = translate_english_to_korean(loc_val)

    # 4. Gemini Question (Interactive)
    ai_message = ""
    if not start_date_val or (not time_val and not is_allday_val):
        current_data = {'date': start_date_val, 'time': time_val, 'loc': loc_val}
        ai_message = ask_gemini_for_missing_info(original_text, current_data, lang=request.lang)

    return ExtractResponse(
        original_text=original_text, translated_text=translated_text,
        summary=summary_val, start_date=start_date_val, end_date=end_date_val,
        time=time_val, location=loc_val, is_allday=is_allday_val,
        ai_message=ai_message, used_model=used_model
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
    except:
        return JSONResponse(status_code=400, content={"error": "Login failed"})


@app.get('/logout')
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url='/')


@app.get('/user-info')
async def get_user_info(request: Request): return {"user": request.session.get('user')}


# [CRUD] Create
@app.post("/add-to-calendar")
async def add_to_calendar(request: Request, event_data: AddEventRequest):
    token_data = request.session.get('token')
    if not token_data: return JSONResponse(status_code=401, content={"error": "Login required"})
    try:
        kst = pytz.timezone('Asia/Seoul')

        # Time Fallback Logic
        s_time = event_data.start_time
        if not s_time and event_data.original_text and not event_data.is_allday:
            fb = dateparser.parse(event_data.original_text, languages=['ko'],
                                  settings={'TIMEZONE': 'Asia/Seoul', 'RETURN_AS_TIMEZONE_AWARE': True})
            if fb: s_time = fb.strftime("%H:%M")

        s_date_obj = dateparser.parse(event_data.start_date, settings={'TIMEZONE': 'Asia/Seoul'})
        e_date_obj = dateparser.parse(event_data.end_date, settings={'TIMEZONE': 'Asia/Seoul'})
        if not s_date_obj: s_date_obj = datetime.datetime.now(kst)
        if not e_date_obj: e_date_obj = s_date_obj

        if event_data.is_allday:
            s_str = s_date_obj.strftime("%Y-%m-%d")
            e_date_excl = e_date_obj + datetime.timedelta(days=1)
            e_str = e_date_excl.strftime("%Y-%m-%d")
            google_event = {'summary': event_data.summary, 'location': event_data.location,
                            'description': event_data.description, 'start': {'date': s_str}, 'end': {'date': e_str}}
        else:
            start_full = f"{event_data.start_date} {s_time}".strip()
            start_dt = dateparser.parse(start_full, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
                                                              'RETURN_AS_TIMEZONE_AWARE': True})
            if not start_dt: start_dt = datetime.datetime.now(kst) + datetime.timedelta(hours=1)

            if event_data.end_time:
                end_full = f"{event_data.end_date} {event_data.end_time}".strip()
                end_dt = dateparser.parse(end_full, settings={'TIMEZONE': 'Asia/Seoul', 'TO_TIMEZONE': 'Asia/Seoul',
                                                              'RETURN_AS_TIMEZONE_AWARE': True})
            else:
                end_dt = start_dt + datetime.timedelta(hours=1)

            google_event = {'summary': event_data.summary, 'location': event_data.location,
                            'description': event_data.description,
                            'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
                            'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Seoul'}}

        access_token = token_data['access_token']
        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        async with httpx.AsyncClient() as client:
            resp = await client.post('https://www.googleapis.com/calendar/v3/calendars/primary/events',
                                     json=google_event, headers=headers)

        if resp.status_code != 200: return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

        if event_data.consent: save_feedback_to_hub(event_data.original_text, event_data.translated_text, event_data)
        return {"message": "Success", "link": resp.json().get('htmlLink'), "saved_msg": "Saved"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# [CRUD] Read
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
                                    params={'timeMin': now, 'maxResults': 50, 'singleEvents': True,
                                            'orderBy': 'startTime'})
        if resp.status_code != 200: return JSONResponse(status_code=resp.status_code, content={"error": "Failed"})
        items = resp.json().get('items', [])
        calendar_events = []
        for event in items:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            calendar_events.append(
                {'id': event['id'], 'title': event.get('summary', 'No Title'), 'start': start, 'end': end,
                 'allDay': 'date' in event['start'],
                 'extendedProps': {'description': event.get('description', ''), 'location': event.get('location', '')}})
        return {"events": calendar_events}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# [CRUD] Update
@app.patch("/events/{event_id}")
async def update_event(request: Request, event_id: str, event_data: UpdateEventRequest):
    token_data = request.session.get('token')
    if not token_data: return JSONResponse(status_code=401, content={"error": "Login required"})
    headers = {'Authorization': f'Bearer {token_data["access_token"]}', 'Content-Type': 'application/json'}
    body = {"summary": event_data.summary, "location": event_data.location, "description": event_data.description}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.patch(f'https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}',
                                      json=body, headers=headers)
        if resp.status_code != 200: return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
        return {"message": "Updated"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# [CRUD] Delete
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
        return {"message": "Deleted"}
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
        except:
            return HTMLResponse("<h1>❌ Failed to list.</h1>")
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files: return HTMLResponse("<h1>📭 No data.</h1>")
        dfs = []
        for file in csv_files:
            try:
                local_filename = hf_hub_download(repo_id=DATASET_REPO_ID, filename=file, repo_type="dataset",
                                                 token=HF_TOKEN)
                dfs.append(pd.read_csv(local_filename))
            except:
                continue
        if not dfs: return HTMLResponse("<h1>❌ Error loading CSV.</h1>")
        final_df = pd.concat(dfs, ignore_index=True)
        if 'timestamp' in final_df.columns: final_df = final_df.sort_values(by='timestamp', ascending=False)
        return HTMLResponse(
            f"<html><head><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'></head><body><div class='container mt-4'><h1>📊 Feedback Log</h1><p>Total: {len(final_df)}</p>{final_df.to_html(classes='table table-striped', index=False)}</div></body></html>")
    except Exception as e:
        return HTMLResponse(f"<h1>❌ Error: {str(e)}</h1>")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")