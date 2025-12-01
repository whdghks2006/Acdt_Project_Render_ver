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
    if not GEMINI_API_KEY: return None

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    lang_instruction = "in Korean" if lang == 'ko' else "in English"

    prompt = f"""
    You are a smart scheduler assistant. Today is {today}.
    User Input: "{text}"

    Your Task:
    1. Extract schedule details.
       - Summary: Concise title.
       - Date Range: If "20th to 23rd", start=20, end=23. If single day, start=end.
       - Time Range: If "2pm to 4pm", start_time="14:00", end_time="16:00". 
         If only "2pm", start_time="14:00", end_time="".

    2. Generate a follow-up question ONLY if critical info is missing.
       - If Date is missing, ask for Date.
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
    start_time: str
    end_time: str
    location: str
    is_allday: bool
    ai_message: str = ""
    used_model: str = ""


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


# [UPDATED] Model for Updating Event (Added date/time fields)
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
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
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
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# [UPDATED] Update Event Endpoint (Supports Date/Time changes)
@app.patch("/events/{event_id}")
async def update_event(request: Request, event_id: str, event_data: UpdateEventRequest):
    token_data = request.session.get('token')
    if not token_data: return JSONResponse(status_code=401, content={"error": "Login required"})

    headers = {'Authorization': f'Bearer {token_data["access_token"]}', 'Content-Type': 'application/json'}

    # Base body
    body = {
        "summary": event_data.summary,
        "location": event_data.location,
        "description": event_data.description
    }

    # Date Logic (similar to Add)
    try:
        if event_data.start_date:
            s_date_obj = dateparser.parse(event_data.start_date)
            e_date_obj = dateparser.parse(event_data.end_date) if event_data.end_date else s_date_obj

            if event_data.is_allday:
                s_str = s_date_obj.strftime("%Y-%m-%d")
                # For update, ensure end date is inclusive if needed, but Google needs exclusive.
                # Assuming simple update for now.
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
        # If date parsing fails, we just update text fields to avoid crash
        pass

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.patch(f'https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}',
                                      json=body, headers=headers)

        if resp.status_code != 200: return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
        return {"message": "Updated successfully"}
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