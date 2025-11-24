import os
import spacy
import dateparser
import datetime
import pandas as pd  # [New]
from urllib.parse import quote_plus
from transformers import pipeline
from huggingface_hub import HfApi  # [New]

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse, HTMLResponse # HTMLResponse 추가!
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth

# ==============================================================================
# Configuration
# ==============================================================================
NER_MODEL_DIR = "my_ner_model"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ko-en"
# [New] Dataset Configuration
DATASET_REPO_ID = "snowmang/scheduler-feedback-data"
DATA_FILENAME = "feedback_log.csv"

# Secrets
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", "random_secret_string")
# [New] Hugging Face Token for uploading data (Must be in Secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")

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


# [New] Function to save feedback
def save_feedback_to_hub(original_text, translated_text, final_data):
    try:
        if not HF_TOKEN:
            print("⚠️ HF_TOKEN not found. Skipping data logging.")
            return

        # Prepare new data row
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

        # Initialize API
        api = HfApi(token=HF_TOKEN)

        # Pull existing file if possible (Append mode)
        # For simplicity in this demo, we just append to a local file and upload
        # In production, you might want to load the existing CSV first.

        # Local save
        local_path = "/tmp/" + DATA_FILENAME

        # Check if file exists in repo (Simple Logic: just upload new file with timestamp in name to avoid conflict)
        # Or simpler: Upload one consolidated file.
        # Let's use a unique filename per entry for safety in this demo to avoid overwrite issues concurrently
        unique_filename = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(unique_filename, index=False)

        # Upload to Hugging Face Dataset
        api.upload_file(
            path_or_fileobj=unique_filename,
            path_in_repo=unique_filename,
            repo_id=DATASET_REPO_ID,
            repo_type="dataset"
        )
        print(f"✅ Feedback data saved to {DATASET_REPO_ID}")

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
        models["translator"] = pipeline("translation", model=TRANSLATION_MODEL)
        print("✅ Translation Model loaded successfully!")
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


# [New] Added original_text and translated_text to capture full context
class AddEventRequest(BaseModel):
    date_str: str
    time_str: str
    loc_str: str
    event_str: str
    description: str
    original_text: str  # [New]
    translated_text: str  # [New]


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
        dt_str = f"{event_data.date_str} {event_data.time_str}"
        start_dt = dateparser.parse(dt_str, settings={'PREFER_DATES_FROM': 'future'})
        if not start_dt: start_dt = datetime.datetime.now() + datetime.timedelta(hours=1)
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
            json=google_event, token=token_data
        )
        resp.raise_for_status()
        result = resp.json()

        # [New] Save User Feedback for future training (HITL)
        save_feedback_to_hub(
            event_data.original_text,
            event_data.translated_text,
            event_data
        )

        return {"message": "Success", "link": result.get('htmlLink')}

    except Exception as e:
        print(f"Calendar Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==============================================================================
# [New] Admin Dashboard Endpoint
# ==============================================================================
@app.get("/admin", response_class=HTMLResponse)  # HTMLResponse 필요
async def admin_dashboard(request: Request):
    # 비밀번호 체크 (간단하게 쿼리 파라미터로 구현)
    # 실제 서비스에선 더 강력한 보안이 필요하지만, 데모용으로는 충분합니다.
    # 사용법: /admin?key=1234
    key = request.query_params.get("key")
    if key != "1234":  # 원하는 비밀번호로 바꾸세요
        return HTMLResponse("<h1>🚫 Access Denied</h1><p>Incorrect admin key.</p>", status_code=403)

    try:
        if not HF_TOKEN:
            return HTMLResponse("<h1>⚠️ HF_TOKEN not set. Cannot fetch data.</h1>")

        # Hugging Face 데이터셋에서 파일 목록 가져오기
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(repo_id=DATASET_REPO_ID, repo_type="dataset")

        # CSV 파일만 필터링
        csv_files = [f for f in files if f.endswith('.csv')]

        if not csv_files:
            return HTMLResponse("<h1>📭 No feedback data found yet.</h1>")

        # 모든 CSV 다운로드 및 병합 (최근 데이터 순)
        dfs = []
        for file in csv_files:
            # 캐시 없이 직접 다운로드 URL 사용 (가장 간단한 방법)
            url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/{file}"
            # Pandas로 읽기 (storage_options로 인증 토큰 전달)
            try:
                df = pd.read_csv(url, storage_options={"token": HF_TOKEN})
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

        if not dfs:
            return HTMLResponse("<h1>❌ Error loading CSV files.</h1>")

        final_df = pd.concat(dfs, ignore_index=True)

        # 최신순 정렬 (timestamp 기준)
        if 'timestamp' in final_df.columns:
            final_df = final_df.sort_values(by='timestamp', ascending=False)

        # HTML Table로 변환
        table_html = final_df.to_html(classes="table table-striped", index=False)

        # 예쁜 디자인을 위한 HTML 템플릿
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Dashboard - AI Scheduler</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            <style>
                body {{ padding: 20px; background-color: #f8f9fa; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: #0d6efd; margin-bottom: 20px; }}
                .badge {{ font-size: 1rem; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="d-flex justify-content-between align-items-center">
                    <h1>📊 Feedback Data Log</h1>
                    <span class="badge bg-success">Total Records: {len(final_df)}</span>
                </div>
                <p>This data is collected from user corrections (Human-in-the-Loop) for future fine-tuning.</p>
                <div class="table-responsive">
                    {table_html}
                </div>
                <hr>
                <a href="/" class="btn btn-secondary">🏠 Back to App</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        return HTMLResponse(f"<h1>❌ System Error: {str(e)}</h1>")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")