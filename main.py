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
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- 3. File Analysis (Text Files) ---
@app.post("/extract-file", response_model=ExtractResponse)
async def api_extract_file_schedule(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        text_content = contents.decode('utf-8')
    except UnicodeDecodeError:
        text_content = contents.decode('euc-kr', errors='ignore')

    gemini_data = extract_info_with_gemini_json(text_content)

    if gemini_data:
        return ExtractResponse(
            original_text="[File Analysis]",
            translated_text="[File Analysis]",
            summary=gemini_data.get("summary", ""),
            start_date=gemini_data.get("start_date", ""),
            end_date=gemini_data.get("end_date", ""),
            start_time=gemini_data.get("start_time") or gemini_data.get("time") or "",
            end_time=gemini_data.get("end_time", ""),
            location=gemini_data.get("location", ""),
            is_allday=gemini_data.get("is_allday", False),
            ai_message=gemini_data.get("question", ""),
            used_model="Gemini 2.5 Flash (File)",
            spacy_log="Skipped (File)"
        )
    else:
        return JSONResponse(status_code=500, content={"error": "File analysis failed"})


# ==============================================================================
# Auth & Calendar CRUD
# ==============================================================================

@app.get('/login')
async def login(request: Request):
    fixed_redirect_uri = "https://snowmang-ai-scheduler-g14.hf.space/auth/callback"
    return await oauth.google.authorize_redirect(request, fixed_redirect_uri)


@app.get('/auth/callback')
async def auth(request: Request):
    try:
        # [Refactor] Removed redirect_uri as per user's working reference
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        request.session['user'] = {'name': user_info.get('name'), 'email': user_info.get('email')}
        request.session['token'] = {'access_token': token.get('access_token'), 'token_type': token.get('token_type')}
        return RedirectResponse(url='/', status_code=303)
    except Exception as e:
        print(f"❌ Auth Error: {e!r}")
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