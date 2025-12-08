# -*- coding: utf-8 -*-
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

# [NEW] File format parsers
import fitz  # PyMuPDF - faster and more accurate than PyPDF2
from docx import Document

from deep_translator import GoogleTranslator
from huggingface_hub import HfApi, hf_hub_download
from pattern_matcher import SchedulePatternMatcher  # [NEW] 커스텀 패턴 매처

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
# Custom NER Model with 6 entity labels
# Labels: START_DATE, START_TIME, END_DATE, END_TIME, LOC, EVENT_TITLE
CUSTOM_NER_MODEL_PATH = "./output/new_ner_model"
FALLBACK_NER_MODEL = "en_core_web_md"  # Fallback if custom model not found
DATASET_REPO_ID = "snowmang/scheduler-feedback-data"

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY", "random_secret_string")
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

models = {}

print(f"?? GEMINI_API_KEY Loaded: {bool(GEMINI_API_KEY)}")
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
        print(f"?? {target_model} failed: {error_msg}")

        # 429 Quota Error는 모델 문제가 아니므로 즉시 에러 반환 (재시도 X)
        if "429" in error_msg or "Resource exhausted" in error_msg:
            raise HTTPException(status_code=429, detail="Google AI Quota Exceeded. Please try again in 1 min.")

        # 그 외 에러(모델 없음 등)는 1.5로 폴백
        print(f"?? Falling back to gemini-1.5-flash...")
        fallback_model = genai.GenerativeModel("gemini-1.5-flash")
        if image:
            return fallback_model.generate_content([prompt, image])
        return fallback_model.generate_content(prompt)


# ==============================================================================
# File Format Parsers (PDF, DOCX, XLSX)
# ==============================================================================

def extract_text_from_pdf(file_bytes):
    """PDF에서 텍스트 추출 (PyMuPDF 사용 - 빠른 속도)"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"❌ PDF Extraction Error: {e}")
        return ""


def clean_pdf_text(text: str) -> str:
    """PDF 텍스트에서 특수 유니코드 문자 제거 (분석 정확도 향상)"""
    import unicodedata
    
    # 제로폭 공백 및 특수 유니코드 제거
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    # 특수 불릿 포인트 정규화
    text = re.sub(r'[●○■□◆◇▶►▪▫•·]', '•', text)
    # 연속 공백 정리
    text = re.sub(r'[ \t]+', ' ', text)
    # 연속 줄바꿈 정리 (3개 이상 -> 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def analyze_assignment_pdf(text: str, nlp_model=None, pattern_matcher=None):
    """
    과제물/실라버스 PDF 전용 분석기 (v4 - spaCy 우선)
    
    핵심 전략:
    1. 텍스트 전처리 (특수 문자 제거)
    2. spaCy + Pattern Matcher로 최대한 추출
    3. Gemini는 사용하지 않음 (호출하는 쪽에서 결정)
    
    Returns:
        list: 추출된 일정 목록 (빈 리스트일 수 있음)
    """
    # 1단계: 텍스트 전처리
    cleaned_text = clean_pdf_text(text)
    print(f"[PDF] Cleaned text: {len(text)} -> {len(cleaned_text)} chars")
    
    schedules = []
    
    # 2단계: Pattern Matcher로 모든 날짜/시간 먼저 추출
    all_dates = []
    all_times = []
    if pattern_matcher:
        all_dates = pattern_matcher.extract_dates(cleaned_text)
        all_times = pattern_matcher.extract_times(cleaned_text)
        print(f"[PDF] Pattern Matcher found: {len(all_dates)} dates, {len(all_times)} times")
    
    # 3단계: "Submit to ... Due:" 블록 패턴으로 분석
    # 전처리된 텍스트에서 더 유연한 패턴 사용
    submit_due_pattern = re.compile(
        r'Submit\s+to\s+([A-Za-z\s&]+?)\s*\([^)]*\)\s*'
        r'Due[:\s]*(?:BY\s*)?(\d{1,2}:\d{2}\s*(?:AM|PM))\s+on\s+'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2})',
        re.IGNORECASE
    )
    
    matches = list(submit_due_pattern.finditer(cleaned_text))
    print(f"[PDF] Submit-Due patterns found: {len(matches)}")
    
    # 4단계: 매치된 패턴 처리
    for match in matches:
        location = match.group(1).strip()
        time_str = match.group(2)
        date_str = match.group(3)
        
        schedule = {
            "summary": "",
            "description": "",
            "start_date": "",
            "end_date": "",
            "start_time": "",
            "end_time": "",
            "location": location,
            "is_allday": False
        }
        
        # 날짜 파싱
        if pattern_matcher and date_str:
            parsed = pattern_matcher.extract_dates(date_str)
            if parsed:
                schedule["start_date"] = parsed[0]
                schedule["end_date"] = parsed[0]
        
        # 시간 파싱
        if pattern_matcher and time_str:
            parsed = pattern_matcher.extract_times(time_str)
            if parsed:
                schedule["end_time"] = parsed[0]
        
        # 과제명 추출 (마감일 앞뒤 컨텍스트)
        match_start = match.start()
        match_end = match.end()
        context_before = cleaned_text[max(0, match_start - 600):match_start]
        context_after = cleaned_text[match_end:min(len(cleaned_text), match_end + 200)]
        
        task_name = ""
        
        # 블랙리스트: 과제명이 아닌 문구들
        blacklist = [
            'academic integrity', 'late submission', 'policy will be applied',
            'turnitin report', 'ai writing', 'detection', 'gptZero', 'scribbr',
            'one thing', 'what you', 'how to', 'will be', 'should be', 'must be',
            'the', 'and', 'for', 'all', 'use', 'is', 'are', 'was', 'were'
        ]
        
        def is_valid_task_name(name):
            """과제명으로 적합한지 검증"""
            if not name or len(name) < 4:
                return False
            name_lower = name.lower()
            for bad in blacklist:
                if bad in name_lower:
                    return False
            # 짧은 일반 단어 제외
            if len(name) < 8 and name_lower in ['poster', 'video', 'essay', 'report']:
                return True  # 이건 허용 (단독 키워드)
            return True
        
        # 전략 A: 앞쪽 컨텍스트에서 과제 키워드 패턴 먼저 검색 (가장 정확)
        keyword_patterns = [
            r'(\d+[-–]\d+\s+page\s+[A-Za-z\s]+?)(?:\s+of|\s*:|\n)',  # 1-2 page Executive Summary
            r'(Individual\s+[A-Za-z\s]+(?:Essay|Report|Reflection))',
            r'((?:Group|Individual)\s+[A-Za-z\s]+(?:Presentation|Essay|Report|Summary|Video|Poster))',
            r'\b(A0\s+[Pp]oster|Executive\s+Summary|Presentation\s+[Vv]ideo)',
            r'(Final\s+Reflection[^.]{0,20})',
            r'(Peer\s+Evaluation[^.]{0,20})',
            r'(Group\s+Capstone\s+[A-Za-z\s]+)',
            r'(Scientific\s+(?:Reflection\s+)?Essay)',
        ]
        for pat in keyword_patterns:
            found = re.findall(pat, context_before, re.IGNORECASE)
            if found:
                candidate = found[-1].strip()
                candidate = re.sub(r'^[\d.\s]+', '', candidate)
                if is_valid_task_name(candidate) and len(candidate) > 5:
                    task_name = candidate
                    print(f"[PDF] Pattern found: '{task_name}'")
                    break
        
        # 전략 B: spaCy NER 사용
        if not task_name and nlp_model:
            doc = nlp_model(context_before[-400:])
            for ent in reversed(list(doc.ents)):
                if ent.label_ in ["EVENT_TITLE", "EVENT"]:
                    candidate = ent.text.strip()[:50]
                    if is_valid_task_name(candidate):
                        task_name = candidate
                        print(f"[PDF] spaCy found: '{task_name}'")
                        break
        
        # 전략 C: 마감일 바로 뒤 항목 (• Poster, • Video) - 필터링 강화
        if not task_name:
            after_match = re.search(r'^[•\s]*([A-Za-z][A-Za-z\s]+?)(?:\s*\(|$|\n)', context_after, re.MULTILINE)
            if after_match:
                potential = after_match.group(1).strip()
                if is_valid_task_name(potential):
                    task_name = potential
        
        # 전략 D: 라인 기반 키워드 검색 (앞쪽)
        if not task_name:
            keywords = ['poster', 'presentation', 'essay', 'video', 'summary', 
                       'report', 'evaluation', 'reflection']
            for line in reversed(context_before.split('\n')[-15:]):
                line = line.strip()
                if 5 < len(line) < 80:
                    for kw in keywords:
                        if kw in line.lower():
                            candidate = re.sub(r'^[\d.•:\s]+', '', line)
                            candidate = re.sub(r'[:.]+$', '', candidate).strip()
                            if is_valid_task_name(candidate) and len(candidate) > 5:
                                task_name = candidate
                                break
                    if task_name:
                        break
        
        # 폴백: 제출처 + 날짜 기반
        if not task_name or len(task_name) < 4:
            task_name = f"Submission to {location}"
        
        # 제목 정리: 앞뒤 특수문자 제거
        task_name = re.sub(r'^[-–—•·\s\d.]+', '', task_name)  # 앞 정리
        task_name = re.sub(r'[-–—•·:.\s]+$', '', task_name)   # 뒤 정리
        task_name = task_name.strip()
        
        # 첫 글자 대문자로
        if task_name and len(task_name) > 1:
            task_name = task_name[0].upper() + task_name[1:]
        
        schedule["summary"] = task_name[:60]
        schedule["is_allday"] = not bool(schedule["end_time"])
        
        schedules.append(schedule)
        print(f"[PDF] ✓ '{schedule['summary']}' -> {schedule['start_date']} {schedule['end_time']} @ {schedule['location']}")
    
    # 5단계: "Week X-Y" 패턴으로 추가 일정 탐색 (Peer Evaluation 등)
    # PDF에서 "Final Reflection & Peer Evaluation (Google Form) • When: Week 15-16" 패턴 감지
    week_pattern = re.compile(
        r'((?:Final\s+)?(?:Reflection|Peer\s+Evaluation|Review)[A-Za-z\s&]*)'
        r'\s*(?:\([^)]*\))?\s*•?\s*When[:\s]*Week\s*(\d+)[-–](\d+)',
        re.IGNORECASE
    )
    
    for week_match in week_pattern.finditer(cleaned_text):
        event_name = week_match.group(1).strip()
        week_start = int(week_match.group(2))
        week_end = int(week_match.group(3))
        
        # 이벤트 이름 정리
        event_name = re.sub(r'^[^A-Za-z]+', '', event_name)  # 앞 쓰레기 제거
        event_name = re.sub(r'[^A-Za-z\s&]+$', '', event_name)  # 뒤 쓰레기 제거
        event_name = event_name.strip()
        
        # 블랙리스트 체크
        if any(bad in event_name.lower() for bad in ['policy', 'late', 'academic', 'integrity']):
            continue
        
        if len(event_name) < 5:
            event_name = "Peer Evaluation"  # 폴백
        
        # Week 15-16 => 대략 12월 중순 (학기 기준)
        import datetime
        base_date = datetime.date(2025, 12, 1)  # Week 14 시작 기준
        start_offset = (week_start - 14) * 7
        end_offset = (week_end - 14) * 7 + 6
        
        start_date = (base_date + datetime.timedelta(days=start_offset)).strftime("%Y-%m-%d")
        end_date = (base_date + datetime.timedelta(days=end_offset)).strftime("%Y-%m-%d")
        
        # 이미 추출된 것과 중복 체크
        existing_summaries = [s["summary"].lower()[:15] for s in schedules]
        if event_name.lower()[:15] not in existing_summaries:
            new_schedule = {
                "summary": event_name[:50],
                "description": f"Week {week_start}-{week_end}",
                "start_date": start_date,
                "end_date": end_date,
                "start_time": "",
                "end_time": "",
                "location": "Google Form" if "evaluation" in event_name.lower() else "",
                "is_allday": True
            }
            schedules.append(new_schedule)
            print(f"[PDF] ✓ (Week pattern) '{new_schedule['summary']}' -> {start_date} ~ {end_date}")
    
    # 6단계: "Week X — Title" 패턴 (날짜 없이 Week만 표시)
    # 예: "Week 9 — Service Concept & Dev Plan", "Week 10 — Data Collection"
    # 더 유연한 패턴: 모든 종류의 대시/구분자 허용
    week_title_pattern = re.compile(
        r'Week\s*(\d+)(?:\s*[-–—]\s*(\d+))?\s*[—–\-:]+\s*([A-Za-z][A-Za-z\s&,]+)',
        re.IGNORECASE
    )
    
    for week_match in week_title_pattern.finditer(cleaned_text):
        week_start = week_match.group(1)
        week_end = week_match.group(2)  # None if single week
        title = week_match.group(3).strip()
        
        # 제목 정리 - "Do this"나 뒤의 설명 제거
        title = re.sub(r'\s*(Do this|Deliverables|Note that|What|●|•).*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'[:\s]+$', '', title).strip()
        # 뒤의 "Weeks" 제거 (예: "Finalization & Showcase Weeks" -> "Finalization & Showcase")
        title = re.sub(r'\s+Weeks?\s*$', '', title, flags=re.IGNORECASE).strip()
        
        # 블랙리스트 체크 - 주요 과제명만 허용
        bad_words = ['project weeks:', 'feedback by', 'what you', 'this is', 'the ']
        if any(bad in title.lower() for bad in bad_words):
            continue
        
        if len(title) < 5 or len(title) > 50:
            continue
        
        # Week 표시
        if week_end:
            week_str = f"Week {week_start}-{week_end}"
        else:
            week_str = f"Week {week_start}"
        
        # 이미 추출된 것과 중복 체크
        existing_summaries = [s["summary"].lower()[:15] for s in schedules]
        if title.lower()[:15] not in existing_summaries:
            new_schedule = {
                "summary": title[:50],
                "description": week_str,
                "start_date": "",
                "end_date": "",
                "start_time": "",
                "end_time": "",
                "location": "",
                "is_allday": True
            }
            schedules.append(new_schedule)
            print(f"[PDF] ✓ (Week title) '{new_schedule['summary']}' ({week_str})")
    
    # 7단계: spaCy로 추가 일정 탐색 (Submit-Due 패턴 외)
    if nlp_model and len(schedules) < 3:
        print("[PDF] Trying additional spaCy-based extraction...")
        doc = nlp_model(cleaned_text[:3000])
        
        for ent in doc.ents:
            if ent.label_ in ["START_DATE", "DATE"]:
                date_text = ent.text
                if pattern_matcher:
                    parsed = pattern_matcher.extract_dates(date_text)
                    if parsed:
                        existing = [s["start_date"] for s in schedules]
                        if parsed[0] not in existing:
                            start_idx = max(0, ent.start_char - 200)
                            end_idx = min(len(cleaned_text), ent.end_char + 100)
                            context = cleaned_text[start_idx:end_idx]
                            
                            event_match = re.search(r'([A-Z][A-Za-z\s]+(?:Essay|Poster|Video|Report|Summary|Presentation|Evaluation))', context)
                            if event_match:
                                new_schedule = {
                                    "summary": event_match.group(1).strip()[:50],
                                    "start_date": parsed[0],
                                    "end_date": parsed[0],
                                    "end_time": "",
                                    "location": "",
                                    "is_allday": True
                                }
                                schedules.append(new_schedule)
                                print(f"[PDF] ✓ (spaCy) '{new_schedule['summary']}' -> {new_schedule['start_date']}")
    
    # 중복 제거
    seen = set()
    unique = []
    for s in schedules:
        # 날짜가 없으면 summary만으로 중복 체크
        if s.get("start_date"):
            key = (s["start_date"], s["summary"][:20].lower())
        else:
            key = ("", s["summary"][:20].lower())
        if key not in seen:
            seen.add(key)
            unique.append(s)
    
    print(f"[PDF] Final: {len(unique)} schedules extracted (spaCy + Pattern Matcher)")
    return unique





def extract_text_from_docx(file_bytes):
    """DOCX에서 텍스트 추출"""
    try:
        docx_file = io.BytesIO(file_bytes)
        doc = Document(docx_file)
        text_parts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n".join(text_parts)
    except Exception as e:
        print(f"? DOCX Extraction Error: {e}")
        return ""

def extract_text_from_xlsx(file_bytes):
    """XLSX에서 텍스트 추출 (모든 셀을 행별로 결합)"""
    try:
        xlsx_file = io.BytesIO(file_bytes)
        df = pd.read_excel(xlsx_file, sheet_name=None)  # 모든 시트 읽기
        text_parts = []
        for sheet_name, sheet_df in df.items():
            # 각 행을 공백으로 결합
            for _, row in sheet_df.iterrows():
                row_text = " ".join([str(val) for val in row.values if pd.notna(val)])
                if row_text.strip():
                    text_parts.append(row_text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"? XLSX Extraction Error: {e}")
        return ""


# ==============================================================================
# AI Logic 1: Text Analysis (spaCy + Gemini Hybrid)
# ==============================================================================
def run_ner_extraction(text, nlp_model, original_text=None):
    """spaCy + Pattern Matcher를 이용한 고정밀 1차 추출"""
    if not text: return "", "", "", ""
    
    # Pattern Matcher는 원본 텍스트 우선 사용 (한국어 패턴 지원)
    text_for_pm = original_text if original_text else text
    
    # Phase 1: Pattern Matcher (높은 정밀도)
    pm_dates, pm_times, pm_loc = [], [], None
    if "pattern_matcher" in models:
        pm = models["pattern_matcher"]
        pm_dates = pm.extract_dates(text_for_pm)
        pm_times = pm.extract_times(text_for_pm)
        pm_loc = pm.extract_locations(text_for_pm)
    
    # Phase 2: spaCy NER (Custom Model with 6 labels)
    doc = nlp_model(text)
    
    # New 6-label extraction
    start_dates = [ent.text for ent in doc.ents if ent.label_ == "START_DATE"]
    start_times = [ent.text for ent in doc.ents if ent.label_ == "START_TIME"]
    end_dates = [ent.text for ent in doc.ents if ent.label_ == "END_DATE"]
    end_times = [ent.text for ent in doc.ents if ent.label_ == "END_TIME"]
    locs = [ent.text for ent in doc.ents if ent.label_ == "LOC" or ent.label_ == "GPE"]
    events = [ent.text for ent in doc.ents if ent.label_ == "EVENT_TITLE" or ent.label_ == "EVENT"]
    
    # Fallback to old labels if custom model labels not found
    if not start_dates:
        start_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    if not start_times:
        start_times = [ent.text for ent in doc.ents if ent.label_ == "TIME"]

    # Phase 3: Merge and validate results
    # Filter spaCy results (remove misidentified entities)
    filtered_times = [t for t in start_times if not _is_date_format(t)]
    filtered_locs = [l for l in locs if not _contains_time_keywords(l)]
    
    # Pattern Matcher takes priority, then filtered spaCy results
    date_str = pm_dates[0] if pm_dates else (", ".join(start_dates) if start_dates else "")
    time_str = pm_times[0] if pm_times else (", ".join(filtered_times) if filtered_times else "")
    loc_str = pm_loc if pm_loc else (", ".join(filtered_locs) if filtered_locs else "")
    
    # End date/time from new labels
    end_date_str = ", ".join(end_dates) if end_dates else ""
    end_time_str = ", ".join(end_times) if end_times else ""

    # Event title inference (Heuristic)
    if events:
        event_str = ", ".join(events)
    elif locs:
        event_str = f"Meeting at {loc_str}"
    else:
        event_str = "New Schedule"

    return date_str, time_str, end_date_str, end_time_str, loc_str, event_str


def extract_multiple_schedules_with_spacy(text, nlp_model):
    """
    Extract multiple schedules using sentence-based spaCy analysis.
    Returns list of schedules or None if extraction quality is low.
    """
    doc = nlp_model(text)
    schedules = []
    
    # Split into sentences
    sentences = list(doc.sents)
    
    for sent in sentences:
        sent_text = sent.text.strip()
        if not sent_text or len(sent_text) < 10:  # Skip very short sentences
            continue
        
        # Re-analyze each sentence
        sent_doc = nlp_model(sent_text)
        
        # Extract entities from sentence
        schedule = {
            "summary": "",
            "description": "",
            "start_date": "",
            "end_date": "",
            "start_time": "",
            "end_time": "",
            "location": "",
            "is_allday": False
        }
        
        has_date_or_time = False
        
        for ent in sent_doc.ents:
            if ent.label_ == "START_DATE" or ent.label_ == "DATE":
                if not schedule["start_date"]:
                    schedule["start_date"] = ent.text
                    has_date_or_time = True
            elif ent.label_ == "START_TIME" or ent.label_ == "TIME":
                if not schedule["start_time"]:
                    schedule["start_time"] = ent.text
                    has_date_or_time = True
            elif ent.label_ == "END_DATE":
                if not schedule["end_date"]:
                    schedule["end_date"] = ent.text
            elif ent.label_ == "END_TIME":
                if not schedule["end_time"]:
                    schedule["end_time"] = ent.text
            elif ent.label_ in ["LOC", "GPE"]:
                if not schedule["location"]:
                    schedule["location"] = ent.text
            elif ent.label_ in ["EVENT_TITLE", "EVENT"]:
                if not schedule["summary"]:
                    schedule["summary"] = ent.text
        
        # Only add if sentence has date or time info
        if has_date_or_time:
            # Generate summary if not found - use smart extraction
            if not schedule["summary"]:
                schedule["summary"] = _extract_smart_summary(sent_text, schedule)
            
            # Set end_date to start_date if not specified
            if schedule["start_date"] and not schedule["end_date"]:
                schedule["end_date"] = schedule["start_date"]
            
            schedules.append(schedule)
    
    # Return None if no schedules found
    if not schedules:
        return None
    
    # If only one schedule found, use single schedule flow
    if len(schedules) == 1:
        return None
    
    # Quality check: if too many schedules have missing info, fall back to Gemini
    quality_score = _calculate_quality_score(schedules)
    if quality_score < 0.5:  # Less than 50% quality
        print(f"[AI] spaCy quality score: {quality_score:.1%} (low) - falling back to Gemini")
        return None
    
    print(f"[AI] spaCy quality score: {quality_score:.1%}")
    return schedules


def _extract_smart_summary(sentence, schedule):
    """Extract a meaningful summary from the sentence."""
    # Common event keywords
    event_keywords = [
        'meeting', 'presentation', 'workshop', 'conference', 'deadline', 
        'appointment', 'interview', 'call', 'session', 'review', 'party',
        'dinner', 'lunch', 'breakfast', 'event', 'ceremony', 'retreat',
        '회의', '미팅', '발표', '워크샵', '마감', '면접', '약속'
    ]
    
    sentence_lower = sentence.lower()
    
    # Find event keyword in sentence
    for keyword in event_keywords:
        if keyword in sentence_lower:
            # Find the context around the keyword
            idx = sentence_lower.find(keyword)
            # Get a few words around it
            start = max(0, sentence.rfind(' ', 0, idx - 10) + 1) if idx > 10 else 0
            end = sentence.find('.', idx)
            if end == -1:
                end = min(len(sentence), idx + 30)
            
            summary = sentence[start:end].strip()
            # Clean up
            summary = summary.split(',')[0].strip()  # Take first part
            if len(summary) > 50:
                summary = summary[:47] + "..."
            return summary.capitalize()
    
    # Fallback: use first meaningful words (skip common starters)
    skip_starters = ['the', 'a', 'an', 'our', 'we', 'i', 'you', 'this', 'that', 'please', 'dear']
    words = sentence.split()
    
    start_idx = 0
    for i, word in enumerate(words[:3]):
        if word.lower() in skip_starters:
            start_idx = i + 1
        else:
            break
    
    summary_words = words[start_idx:start_idx + 4]
    summary = " ".join(summary_words)
    
    if len(summary) > 40:
        summary = summary[:37] + "..."
    
    return summary.capitalize() if summary else "Schedule"


def _calculate_quality_score(schedules):
    """Calculate quality score for extracted schedules (0-1)."""
    if not schedules:
        return 0
    
    total_score = 0
    
    for s in schedules:
        score = 0
        # Must have date OR time
        if s.get("start_date") or s.get("start_time"):
            score += 0.3
        # Meaningful summary (not truncated)
        if s.get("summary") and not s["summary"].endswith("..."):
            score += 0.3
        elif s.get("summary"):
            score += 0.15
        # Has location
        if s.get("location"):
            score += 0.2
        # Both date and time
        if s.get("start_date") and s.get("start_time"):
            score += 0.2
        
        total_score += score
    
    return total_score / len(schedules)


def _is_date_format(text: str) -> bool:
    """Check if text matches date format (e.g., 12/20)"""
    import re
    return bool(re.match(r'^\d{1,2}/\d{1,2}$', text))


def _contains_time_keywords(text: str) -> bool:
    """Check if text contains time-related keywords"""
    time_keywords = ['tomorrow', 'today', 'next', 'this', '내일', '오늘', '다음', '이번']
    return any(keyword in text.lower() for keyword in time_keywords)


def extract_info_with_gemini_json(text):
    """Gemini를 이용한 정밀 2차 추출"""
    if not GEMINI_API_KEY: return None
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 입력 언어 감지
    is_korean_input = check_is_korean(text)
    lang_instruction = "IMPORTANT: The input text is in Korean. You MUST keep the 'summary' and 'description' in Korean." if is_korean_input else ""

    prompt = f"""
    You are a smart scheduler assistant. Today is {today}.
    Extract schedule details from: "{text}"
    {lang_instruction}
    
    (Note: If this looks like a chat log, ignore message timestamps and focus on the conversation content.)

    CRITICAL RULES:
    1. **DO NOT invent or assume dates/times if they are not explicitly mentioned in the text.**
       - If no date is mentioned, return empty string "" for start_date and end_date.
       - If no time is mentioned, return empty string "" for start_time and end_time, and set is_allday to false.
    
    2. Handle date ranges (start_date, end_date). If single day is mentioned, start=end.
    
    3. **Keep the summary SHORT and simple** (max 5-7 words). Put detailed information in the 'description' field.
    
    4. **Preserve the original language**: If input is Korean, summary and description MUST be in Korean.
    
    5. Separate start_time and end_time if a range is given (e.g. 2pm-4pm).
    
    6. **Description field**: Include any additional details, context, or notes about the event here.

    Return JSON ONLY: 
    {{ "summary": "...", "description": "...", "start_date": "YYYY-MM-DD or empty", "end_date": "YYYY-MM-DD or empty", "start_time": "HH:MM or empty", "end_time": "HH:MM or empty", "location": "...", "is_allday": boolean }}
    """
    try:
        # [Updated] Try 2.5 -> Fallback 1.5
        response = get_gemini_content(prompt, target_model='gemini-2.5-flash')
        clean = re.sub(r'```json|```', '', response.text).strip()
        parsed = json.loads(clean)
        
        # Type check: Gemini should return a dict, not a list
        if not isinstance(parsed, dict):
            print(f"[WARN] Gemini returned unexpected type: {type(parsed)}")
            return None
        
        return parsed
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON Parse Error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Gemini Error: {e}")
        return None


def extract_multiple_schedules_with_gemini(text):
    """Gemini를 이용한 여러 일정 추출"""
    if not GEMINI_API_KEY: return None
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    is_korean_input = check_is_korean(text)
    lang_instruction = "IMPORTANT: The input text is in Korean. You MUST keep the 'summary' and 'description' in Korean." if is_korean_input else ""

    prompt = f"""
    You are a smart scheduler assistant. Today is {today}.
    Extract ALL schedules from: "{text}"
    {lang_instruction}
    
    CRITICAL RULES:
    1. **Extract MULTIPLE schedules if mentioned** (e.g., "내일 3시 회의, 모레 5시 저녁" = 2 schedules)
    2. **DO NOT invent dates/times** - if not mentioned, return empty string
    3. Keep summary SHORT (max 5-7 words)
    4. **Preserve original language** for summary and description
    
    Return JSON array of schedules:
    [
        {{"summary": "...", "description": "...", "start_date": "YYYY-MM-DD or empty", "end_date": "YYYY-MM-DD or empty", "start_time": "HH:MM or empty", "end_time": "HH:MM or empty", "location": "...", "is_allday": boolean}},
        ...
    ]
    """
    try:
        response = get_gemini_content(prompt, target_model='gemini-2.5-flash')
        clean = re.sub(r'```json|```', '', response.text).strip()
        parsed = json.loads(clean)
        
        # 배열이어야 함
        if not isinstance(parsed, list):
            print(f"[WARN] Gemini returned non-list: {type(parsed)}")
            # 단일 객체면 배열로 감싸기
            if isinstance(parsed, dict):
                return [parsed]
            return None
        
        return parsed
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON Parse Error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Gemini Error: {e}")
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
        print("? GEMINI_API_KEY is missing.")
        return ""
    
    try:
        image = PIL.Image.open(io.BytesIO(image_bytes))
        prompt = "Transcribe all text from this image exactly as it appears. Do not summarize."
        
        # Try 2.5 -> Fallback 1.5
        response = get_gemini_content(prompt, image=image, target_model='gemini-2.5-flash')
        if not response or not response.text:
            print("? Gemini returned empty response.")
            return ""
        return response.text.strip()
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"? Vision Transcription Error: {e}")
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
        3. **If the text is in Korean, the 'summary' and 'description' MUST be in Korean.**
        
        CRITICAL RULES:
        - **DO NOT invent or assume dates/times if they are not explicitly mentioned.**
        - If no date is mentioned, return empty string "" for start_date and end_date.
        - If no time is mentioned, return empty string "" for start_time and end_time.
        - **Keep summary SHORT** (max 5-7 words). Put details in 'description'.
        - **Preserve original language**: Korean input → Korean output.

        Return JSON ONLY:
        {{
          "summary": "...",
          "description": "...",
          "start_date": "YYYY-MM-DD or empty", "end_date": "YYYY-MM-DD or empty",
          "start_time": "HH:MM or empty", "end_time": "HH:MM or empty",
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
        print(f"? Vision Analysis Error: {error_msg}")
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
        print(f"? Feedback saved.")
    except Exception as e:
        print(f"? Save Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("? App Started")
    # Load NER model on startup (Custom model with fallback)
    try:
        if os.path.exists(CUSTOM_NER_MODEL_PATH):
            models["nlp_sm"] = spacy.load(CUSTOM_NER_MODEL_PATH)
            print(f"? Custom NER model loaded from '{CUSTOM_NER_MODEL_PATH}'")
            print(f"  Labels: START_DATE, START_TIME, END_DATE, END_TIME, LOC, EVENT_TITLE")
        else:
            print(f"?? Custom model not found at '{CUSTOM_NER_MODEL_PATH}'")
            print(f"   Falling back to '{FALLBACK_NER_MODEL}'...")
            if not spacy.util.is_package(FALLBACK_NER_MODEL):
                print(f"?? Downloading {FALLBACK_NER_MODEL}...")
                spacy.cli.download(FALLBACK_NER_MODEL)
            models["nlp_sm"] = spacy.load(FALLBACK_NER_MODEL)
            print(f"? Fallback model '{FALLBACK_NER_MODEL}' loaded.")
    except Exception as e:
        print(f"? Failed to load spaCy: {e}")
    
    # Initialize Pattern Matcher
    try:
        models["pattern_matcher"] = SchedulePatternMatcher()
        print(f"? Custom Pattern Matcher initialized.")
    except Exception as e:
        print(f"? Failed to initialize Pattern Matcher: {e}")
    
    yield
    print("? App Shutdown")


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
    description: str = ""
    is_allday: bool
    ai_message: str = ""
    used_model: str = ""
    spacy_log: str = ""
    schedules: list = []  # 여러 일정 지원


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

    date_str, time_str, end_date_str, end_time_str, loc_str, event_str = "", "", "", "", "", ""
    if "nlp_sm" in models:
        date_str, time_str, end_date_str, end_time_str, loc_str, event_str = run_ner_extraction(translated_text, models["nlp_sm"], original_text)

    spacy_debug_str = f"StartDate=[{date_str}] StartTime=[{time_str}] EndDate=[{end_date_str}] EndTime=[{end_time_str}] Loc=[{loc_str}]"
    # ===== 여러 일정 감지 (하이브리드 방식) =====
    # Pattern Matcher로 빠른 감지
    multiple_dates = []
    multiple_times = []
    if "pattern_matcher" in models:
        pm = models["pattern_matcher"]
        multiple_dates = pm.extract_dates(original_text)
        multiple_times = pm.extract_times(original_text)
    
    # Check for multiple schedules (but exclude time ranges like "10 AM to 4 PM")
    # If " to " or " - " pattern exists with 2 times, it's likely a single schedule with time range
    is_time_range = (len(multiple_times) == 2 and 
                     (" to " in original_text.lower() or " - " in original_text or 
                      "부터" in original_text or "~" in original_text))
    
    # Only treat as multiple schedules if:
    # - More than 2 dates, OR
    # - More than 2 times (not a simple range), OR
    # - 2+ dates with 2+ times
    is_multiple = (len(multiple_dates) > 2 or 
                   (len(multiple_times) > 2) or 
                   (len(multiple_dates) > 1 and len(multiple_times) > 1 and not is_time_range))
    
    if is_multiple:
        print(f"[AI] Multiple schedules detected (dates:{len(multiple_dates)}, times:{len(multiple_times)})")
        
        # Try spaCy first (faster)
        schedules = None
        if "nlp_sm" in models:
            print("[AI] Trying spaCy sentence-based extraction...")
            schedules = extract_multiple_schedules_with_spacy(translated_text, models["nlp_sm"])
        
        # If spaCy succeeded
        if schedules and len(schedules) > 1:
            print(f"[AI] spaCy extracted {len(schedules)} schedules successfully!")
            return ExtractResponse(
                original_text=original_text,
                translated_text=translated_text,
                summary="",
                description="",
                start_date="",
                end_date="",
                start_time="",
                end_time="",
                location="",
                is_allday=False,
                ai_message="",
                used_model="⚡ Fast (spaCy - Multiple)",
                spacy_log=spacy_debug_str,
                schedules=schedules
            )
        
        # Fallback to Gemini if spaCy didn't work
        print("[AI] spaCy insufficient, falling back to Gemini...")
        schedules = extract_multiple_schedules_with_gemini(original_text)
        
        if schedules and len(schedules) > 1:
            return ExtractResponse(
                original_text=original_text,
                translated_text=translated_text,
                summary="",
                description="",
                start_date="",
                end_date="",
                start_time="",
                end_time="",
                location="",
                is_allday=False,
                ai_message="",
                used_model="🧠 Smart (Gemini 2.5 - Multiple)",
                spacy_log=spacy_debug_str,
                schedules=schedules
            )
    
    # ===== 단일 일정 처리 (기존 로직) =====


    # Set Initial Values
    summary_val = original_text if is_korean_input else event_str
    description_val = ""
    start_date_val = date_str
    end_date_val = end_date_str if end_date_str else date_str  # Use NER end_date or fallback to start_date
    start_time_val = time_str
    end_time_val = end_time_str  # Use NER end_time
    loc_val = loc_str
    is_allday_val = False
    used_model = "Fast-Inference (spaCy)"
    ai_message = ""

    # [Branch] If fast mode, return immediately
    if mode == "fast":
        if is_korean_input: loc_val = translate_english_to_korean(loc_val)
        return ExtractResponse(
            original_text=original_text, translated_text=translated_text,
            summary=summary_val, description=description_val,
            start_date=start_date_val, end_date=end_date_val,
            start_time=start_time_val, end_time=end_time_val,
            location=loc_val, is_allday=is_allday_val,
            ai_message="", used_model="⚡ Fast (spaCy)",
            spacy_log=spacy_debug_str
        )

    # 2. Gemini Extraction (Smart Fallback)
    # Only call if spaCy missed critical info
    needs_gemini = is_ocr or not date_str or not time_str
    
    # " to " pattern: only call Gemini if END_TIME wasn't already extracted by spaCy
    if " to " in translated_text and not end_time_str:
        needs_gemini = True
    
    if needs_gemini:
        print("[AI] spaCy incomplete. Calling Gemini...")
        gemini_data = extract_info_with_gemini_json(original_text)

        if gemini_data:
            summary_val = gemini_data.get("summary") or summary_val
            description_val = gemini_data.get("description") or ""
            start_date_val = gemini_data.get("start_date") or ""
            end_date_val = gemini_data.get("end_date") or ""

            g_start = gemini_data.get("start_time") or gemini_data.get("time")
            start_time_val = g_start or ""
            end_time_val = gemini_data.get("end_time") or ""

            loc_val = gemini_data.get("location") or loc_val
            is_allday_val = gemini_data.get("is_allday") or False
            used_model = "🧠 Smart (Gemini 2.5)"

    # 3. Localization
    if is_korean_input and used_model.startswith("Fast"):
        loc_val = translate_english_to_korean(loc_val)

    # 4. Interactive Question
    if not start_date_val or (not start_time_val and not is_allday_val):
        current_data = {'date': start_date_val, 'time': start_time_val, 'loc': loc_val}
        ai_message = ask_gemini_for_missing_info(original_text, current_data, lang=lang)

    return ExtractResponse(
        original_text=original_text, translated_text=translated_text,
        summary=summary_val, description=description_val,
        start_date=start_date_val, end_date=end_date_val,
        start_time=start_time_val, end_time=end_time_val,
        location=loc_val, is_allday=is_allday_val,
        ai_message=ai_message, used_model=used_model,
        spacy_log=spacy_debug_str
    )


@app.post("/extract", response_model=ExtractResponse)
async def api_extract_schedule(request: ExtractRequest):
    return process_text_schedule(request.text, request.mode, request.lang)


# --- 1.5. Quality Enhancement (Force Gemini Re-analysis) ---
class ReanalyzeRequest(BaseModel):
    text: str
    lang: str = 'en'

@app.post("/reanalyze-gemini", response_model=ExtractResponse)
async def api_reanalyze_with_gemini(request: ReanalyzeRequest):
    """Force re-analysis using Gemini for better quality"""
    original_text = request.text
    is_korean_input = check_is_korean(original_text)
    translated_text = translate_korean_to_english(original_text) if is_korean_input else original_text
    
    print("[AI] Quality Enhancement: Force Gemini re-analysis")
    
    # Check for multiple schedules first
    schedules = extract_multiple_schedules_with_gemini(original_text)
    
    if schedules and len(schedules) > 1:
        return ExtractResponse(
            original_text=original_text,
            translated_text=translated_text,
            summary="",
            description="",
            start_date="",
            end_date="",
            start_time="",
            end_time="",
            location="",
            is_allday=False,
            ai_message="",
            used_model="🧠 Smart (Gemini 2.5 - Enhanced)",
            spacy_log="Quality Enhancement Mode",
            schedules=schedules
        )
    
    # Single schedule - use Gemini extraction
    gemini_data = extract_info_with_gemini_json(original_text)
    
    if gemini_data:
        return ExtractResponse(
            original_text=original_text,
            translated_text=translated_text,
            summary=gemini_data.get("summary", ""),
            description=gemini_data.get("description", ""),
            start_date=gemini_data.get("start_date", ""),
            end_date=gemini_data.get("end_date", ""),
            start_time=gemini_data.get("start_time", ""),
            end_time=gemini_data.get("end_time", ""),
            location=gemini_data.get("location", ""),
            is_allday=gemini_data.get("is_allday", False),
            ai_message="",
            used_model="🧠 Smart (Gemini 2.5 - Enhanced)",
            spacy_log="Quality Enhancement Mode"
        )
    
    # Fallback if Gemini fails
    return ExtractResponse(
        original_text=original_text,
        translated_text=translated_text,
        summary="Analysis failed",
        description="",
        start_date="",
        end_date="",
        start_time="",
        end_time="",
        location="",
        is_allday=False,
        ai_message="Gemini analysis failed. Please try again.",
        used_model="Error",
        spacy_log=""
    )


# --- 2. Image Analysis (Vision) ---
@app.post("/extract-image", response_model=ExtractResponse)
async def api_extract_image_schedule(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # 1. Transcribe Image to Text
        transcribed_text = run_vision_transcription(contents)
        
        if not transcribed_text:
            return JSONResponse(status_code=500, content={"error": "Failed to read text from image."})
            
        print(f"?? Image Transcribed: {transcribed_text[:50]}...")

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


# --- 3. File Analysis (Enhanced: TXT, PDF, DOCX) ---
@app.post("/extract-file", response_model=ExtractResponse)
async def api_extract_file_schedule(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()
    
    # Determine file type and extract text
    text_content = ""
    
    if filename.endswith('.pdf'):
        print("[PDF] Processing PDF file...")
        text_content = extract_text_from_pdf(contents)
        
        # PDF 전용 분석: 커스텀 NER + Pattern Matcher 우선 사용
        if text_content and "nlp_sm" in models:
            print("[PDF] Step 1: Using spaCy + Pattern Matcher...")
            nlp_model = models.get("nlp_sm")
            pm = models.get("pattern_matcher")
            
            # spaCy + Pattern Matcher로 분석
            spacy_schedules = analyze_assignment_pdf(text_content, nlp_model, pm)
            spacy_count = len(spacy_schedules) if spacy_schedules else 0
            
            # 결과가 충분하면 (3개 이상) spaCy 결과만 반환
            if spacy_schedules and spacy_count >= 3:
                print(f"[PDF] ✓ spaCy extraction successful: {spacy_count} schedules")
                if spacy_count > 1:
                    return ExtractResponse(
                        original_text=text_content[:500] + "..." if len(text_content) > 500 else text_content,
                        translated_text="",
                        summary="",
                        description="",
                        start_date="",
                        end_date="",
                        start_time="",
                        end_time="",
                        location="",
                        is_allday=False,
                        ai_message="",
                        used_model="⚡ spaCy + Pattern Matcher (Fast)",
                        spacy_log=f"Extracted {spacy_count} schedules using custom NER model",
                        schedules=spacy_schedules
                    )
                else:
                    s = spacy_schedules[0]
                    return ExtractResponse(
                        original_text=text_content[:500] + "..." if len(text_content) > 500 else text_content,
                        translated_text="",
                        summary=s.get("summary", ""),
                        description=s.get("description", ""),
                        start_date=s.get("start_date", ""),
                        end_date=s.get("end_date", ""),
                        start_time=s.get("start_time", ""),
                        end_time=s.get("end_time", ""),
                        location=s.get("location", ""),
                        is_allday=s.get("is_allday", False),
                        ai_message="",
                        used_model="⚡ spaCy + Pattern Matcher (Fast)",
                        spacy_log="Single schedule extracted with custom NER"
                    )
            
            # 결과가 부족하면 (3개 미만) Gemini로 보완
            print(f"[PDF] Step 2: spaCy found {spacy_count} schedules, trying Gemini enhancement...")
            
            # Gemini로 추가 분석
            gemini_schedules = extract_multiple_schedules_with_gemini(text_content)
            gemini_count = len(gemini_schedules) if gemini_schedules else 0
            
            if gemini_schedules and gemini_count > spacy_count:
                print(f"[PDF] ✓ Gemini enhanced: {gemini_count} schedules (was {spacy_count})")
                
                # spaCy 결과가 있었다면 로그에 표시
                spacy_log_msg = f"spaCy found {spacy_count}, Gemini enhanced to {gemini_count} schedules"
                
                if gemini_count > 1:
                    return ExtractResponse(
                        original_text=text_content[:500] + "..." if len(text_content) > 500 else text_content,
                        translated_text="",
                        summary="",
                        description="",
                        start_date="",
                        end_date="",
                        start_time="",
                        end_time="",
                        location="",
                        is_allday=False,
                        ai_message="",
                        used_model="🔄 spaCy → Gemini (Enhanced)",
                        spacy_log=spacy_log_msg,
                        schedules=gemini_schedules
                    )
                else:
                    s = gemini_schedules[0]
                    return ExtractResponse(
                        original_text=text_content[:500] + "..." if len(text_content) > 500 else text_content,
                        translated_text="",
                        summary=s.get("summary", ""),
                        description=s.get("description", ""),
                        start_date=s.get("start_date", ""),
                        end_date=s.get("end_date", ""),
                        start_time=s.get("start_time", ""),
                        end_time=s.get("end_time", ""),
                        location=s.get("location", ""),
                        is_allday=s.get("is_allday", False),
                        ai_message="",
                        used_model="🔄 spaCy → Gemini (Enhanced)",
                        spacy_log=spacy_log_msg
                    )
            
            # spaCy 결과라도 있으면 반환
            elif spacy_schedules and spacy_count > 0:
                print(f"[PDF] Using spaCy results ({spacy_count} schedules)")
                if spacy_count > 1:
                    return ExtractResponse(
                        original_text=text_content[:500] + "..." if len(text_content) > 500 else text_content,
                        translated_text="",
                        summary="",
                        description="",
                        start_date="",
                        end_date="",
                        start_time="",
                        end_time="",
                        location="",
                        is_allday=False,
                        ai_message="",
                        used_model="⚡ spaCy + Pattern Matcher",
                        spacy_log=f"Extracted {spacy_count} schedules (Gemini unavailable)",
                        schedules=spacy_schedules
                    )
                else:
                    s = spacy_schedules[0]
                    return ExtractResponse(
                        original_text=text_content[:500] + "..." if len(text_content) > 500 else text_content,
                        translated_text="",
                        summary=s.get("summary", ""),
                        description=s.get("description", ""),
                        start_date=s.get("start_date", ""),
                        end_date=s.get("end_date", ""),
                        start_time=s.get("start_time", ""),
                        end_time=s.get("end_time", ""),
                        location=s.get("location", ""),
                        is_allday=s.get("is_allday", False),
                        ai_message="",
                        used_model="⚡ spaCy + Pattern Matcher",
                        spacy_log="Single schedule extracted"
                    )
            
            print("[PDF] No schedules found by either method, falling back to general processing...")

        
    elif filename.endswith('.docx'):
        print("[DOCX] Processing DOCX file...")
        text_content = extract_text_from_docx(contents)
    elif filename.endswith('.txt'):
        print("[TXT] Processing TXT file...")
        try:
            text_content = contents.decode('utf-8')
        except UnicodeDecodeError:
            text_content = contents.decode('euc-kr', errors='ignore')
    else:
        return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {filename}"})
    
    if not text_content or not text_content.strip():
        return JSONResponse(status_code=500, content={"error": "Failed to extract text from file"})
    
    print(f"[OK] Extracted {len(text_content)} characters from {filename}")
    
    # Use the unified text processing pipeline (spaCy + Gemini)
    # Force Gemini usage for file analysis to ensure high-quality extraction
    result = process_text_schedule(text_content, mode="full", is_ocr=True)
    
    # Create a new response with updated model name (Pydantic models are immutable)
    file_type = filename.split('.')[-1].upper()
    return ExtractResponse(
        original_text=result.original_text,
        translated_text=result.translated_text,
        summary=result.summary,
        start_date=result.start_date,
        end_date=result.end_date,
        start_time=result.start_time,
        end_time=result.end_time,
        location=result.location,
        is_allday=result.is_allday,
        ai_message=result.ai_message,
        description=result.description,
        used_model=f"File ({file_type}) + {result.used_model}",
        spacy_log=result.spacy_log,
        schedules=result.schedules
    )



# ==============================================================================
# Auth & Calendar CRUD
# ==============================================================================

@app.get('/login')
async def login(request: Request):
    # 현재 환경(로컬/Hugging Face)에 맞게 자동으로 redirect_uri 생성
    redirect_uri = request.url_for('auth')
    
    # Hugging Face Spaces는 리버스 프록시 뒤에서 실행되므로 HTTPS로 변경 필요
    # X-Forwarded-Proto 헤더가 있으면 HTTPS 환경
    if request.headers.get('x-forwarded-proto') == 'https':
        redirect_uri = str(redirect_uri).replace('http://', 'https://')
    
    print(f" Generated redirect_uri: {redirect_uri}")
    return await oauth.google.authorize_redirect(request, redirect_uri)


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
        print(f"? Auth Error: {e!r}")
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
            saved_msg = "? Data saved."
        else:
            saved_msg = "?? Data NOT saved."

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
    now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')

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
    if key != "1234": return HTMLResponse("<h1>?? Access Denied</h1>", status_code=403)

    try:
        if not HF_TOKEN: return HTMLResponse("<h1>?? HF_TOKEN not set.</h1>")
        api = HfApi(token=HF_TOKEN)
        try:
            files = api.list_repo_files(repo_id=DATASET_REPO_ID, repo_type="dataset")
        except Exception as e:
            return HTMLResponse(f"<h1>? Failed to list files.</h1><pre>{str(e)}</pre>")

        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files: return HTMLResponse("<h1>?? No data found.</h1>")

        dfs = []
        for file in csv_files:
            try:
                local_filename = hf_hub_download(repo_id=DATASET_REPO_ID, filename=file, repo_type="dataset",
                                                 token=HF_TOKEN)
                df = pd.read_csv(local_filename)
                dfs.append(df)
            except Exception:
                continue

        if not dfs: return HTMLResponse("<h1>? Error loading CSV.</h1>")
        final_df = pd.concat(dfs, ignore_index=True)
        if 'timestamp' in final_df.columns: final_df = final_df.sort_values(by='timestamp', ascending=False)
        table_html = final_df.to_html(classes="table table-striped", index=False)
        return HTMLResponse(
            f"<html><head><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'></head><body><div class='container mt-4'><h1>?? Feedback Log</h1><p>Total: {len(final_df)}</p>{table_html}</div></body></html>")

    except Exception as e:
        return HTMLResponse(f"<h1>? Error: {str(e)}</h1>")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")


