# INPUT 개선 개발 계획서 (최종본)

## 목표

교수님 피드백에 따라 **INPUT의 접근성과 편의성을 향상**시키면서, 동시에 **처리 속도를 개선**합니다.

### 핵심 원칙
- ✅ **spaCy 모델 활용 + Gemini 정확성 향상** 방식은 **반드시 유지** (프로젝트의 핵심)
- ✅ 현재 구동 방식에서 큰 차이 없도록 개선
- ✅ 실용적이고 즉시 사용 가능한 기능 우선

---

## 승인된 개선 사항 (4가지)

### 1. 클립보드 자동 감지 (Smart Clipboard Detection)

**문제점:** 사용자가 이메일, 메신저 등에서 일정 정보를 복사한 후 수동으로 붙여넣기 해야 함

**해결책:**
- 페이지 방문 시 클립보드에 일정 관련 내용이 있으면 자동 감지
- "클립보드에서 일정을 감지했습니다. 분석하시겠습니까?" 알림 표시
- 원클릭으로 즉시 분석 실행

**기술 스택:**
- Clipboard API (JavaScript)
- 간단한 패턴 매칭으로 날짜/시간 키워드 감지

**예상 효과:** 복사-붙여넣기 2단계 → 1단계로 단축

---

### 2. 파일 형식 확장 (Enhanced File Format Support)

**현재:** `.txt` 텍스트 파일만 지원

**추가 지원 형식:**
- **PDF** - 이메일/문서 첨부 파일에서 일정 추출
- **DOCX** - Word 문서 지원
- **XLSX** - Excel 스프레드시트 (표 형식 일정 데이터)

**구현 방법:**
- PDF: `PyPDF2` 또는 `pdfplumber`로 텍스트 추출 후 기존 파이프라인 활용
- DOCX: `python-docx`로 텍스트 추출
- XLSX: `pandas`로 표 데이터 읽기 후 행별 분석

**예상 효과:** 이메일 첨부파일, 공식 문서를 직접 업로드하여 분석 가능

---

### 3. 빠른 입력 템플릿 (Quick Input Templates)

**문제점:** 반복되는 일정(예: 매주 팀 미팅)을 매번 새로 입력해야 함

**해결책:**
- 자주 사용하는 일정을 템플릿으로 저장
- 템플릿 선택 시 자동으로 입력 필드 채우기
- 로컬 스토리지에 저장 (서버 부담 없음)

**주요 기능:**
- 템플릿 저장: "💾 템플릿으로 저장" 버튼
- 템플릿 불러오기: 드롭다운에서 선택 → 자동 입력
- 템플릿 관리: 이름 변경, 삭제

**예상 효과:** 반복 작업 시간 80% 단축

---

### 4. 속도 최적화 (Performance Optimization)

**현재 속도 이슈:**
- Gemini API 호출로 인한 지연 (2-5초)
- 번역 API 중복 호출
- 불필요한 Gemini 호출

**개선 방안 (spaCy+Gemini 방식 유지):**

#### A. 병렬 처리 (Parallel Processing)
- spaCy NER 추출과 번역을 동시 실행
- 현재: 순차적 → 개선: 비동기 병렬

#### B. 캐싱 시스템 (Caching)
- 동일한 텍스트 재분석 시 캐시 활용
- 번역 결과 캐싱 (같은 문장 재번역 방지)
- 메모리 기반 LRU 캐시 (최근 100개 결과 저장)

#### C. 스마트 Gemini 호출 최적화
- **현재 로직 유지**: spaCy가 날짜/시간을 못 찾은 경우만 Gemini 호출
- **추가 최적화**: spaCy가 충분한 정보를 찾으면 Gemini 스킵 (사용자 선택 가능)
- "⚡ 빠른 모드" vs "🎯 정확 모드" 토글 추가

#### D. 프론트엔드 최적화
- 입력 중 실시간 디바운싱 (0.5초 대기 후 분석)
- 로딩 상태 개선 (프로그레스 바, 예상 시간 표시)
- 분석 결과 즉시 표시 (부분 결과 우선 렌더링)

**핵심: spaCy 모델 활용은 항상 1순위로 실행되며, Gemini는 보조 역할 유지**

**예상 효과:** 평균 응답 시간 50% 단축 (5초 → 2.5초)

---

## Proposed Changes

### Component 1: Clipboard Detection System

#### [NEW] [static/clipboard-monitor.js](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/static/clipboard-monitor.js)
- Clipboard API를 사용한 자동 감지 로직
- 날짜/시간 패턴 매칭 (정규식)
- 알림 배너 UI 컨트롤

#### [MODIFY] [static/index.html](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/static/index.html)
- 클립보드 알림 배너 추가 (상단 고정)
- "📋 클립보드에서 가져오기" 버튼
- clipboard-monitor.js 스크립트 로드

---

### Component 2: Enhanced File Format Support

#### [MODIFY] [main.py](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/main.py)

**변경 내용:**
- `api_extract_file_schedule` 함수 확장
- 파일 확장자 감지 및 적절한 파서 호출
- PDF/DOCX/XLSX 텍스트 추출 로직 추가

**새로운 헬퍼 함수:**
```python
def extract_text_from_pdf(file_bytes):
    """PDF에서 텍스트 추출"""
    
def extract_text_from_docx(file_bytes):
    """DOCX에서 텍스트 추출"""
    
def extract_text_from_xlsx(file_bytes):
    """XLSX에서 텍스트 추출 (행별 결합)"""
```

#### [MODIFY] [requirements.txt](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/requirements.txt)

**추가 패키지:**
- `PyPDF2` - PDF 파싱
- `python-docx` - Word 문서 파싱
- `openpyxl` - Excel 파싱 (pandas가 내부적으로 사용)

#### [MODIFY] [static/index.html](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/static/index.html)
- 파일 입력 accept 속성 확장: `.txt, .pdf, .docx, .xlsx`
- 드롭존 안내 문구 업데이트

---

### Component 3: Quick Input Templates

#### [NEW] [static/template-manager.js](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/static/template-manager.js)

**기능:**
- localStorage를 사용한 템플릿 CRUD
- JSON 형식으로 템플릿 저장
- 템플릿 목록 렌더링 및 선택 이벤트 처리

**템플릿 데이터 구조:**
```javascript
{
  id: "template_123",
  name: "매주 팀 미팅",
  data: {
    summary: "팀 미팅",
    start_date: "2025-12-05",
    start_time: "14:00",
    location: "회의실 A",
    // ...
  },
  created: "2025-12-03T17:00:00"
}
```

#### [MODIFY] [static/index.html](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/static/index.html)

**UI 추가:**
- 우측 컬럼 상단에 "💾 템플릿으로 저장" 버튼
- 템플릿 드롭다운 (상단 고정)
- 템플릿 관리 모달 (목록, 삭제, 이름 변경)

---

### Component 4: Performance Optimization

#### [MODIFY] [main.py](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/main.py)

**A. 캐싱 시스템 추가:**
```python
from functools import lru_cache

# 번역 캐싱
@lru_cache(maxsize=100)
def translate_korean_to_english_cached(text):
    return translate_korean_to_english(text)

# 일정 분석 결과 캐싱 (텍스트 해시 기반)
analysis_cache = {}
```

**B. 비동기 병렬 처리:**
- spaCy NER과 번역을 동시 실행 (현재는 순차적)
- `asyncio.gather()`를 사용한 병렬 처리

**C. 스마트 Gemini 호출:**
- spaCy가 충분한 정보를 찾은 경우 Gemini 스킵 옵션 추가
- `mode` 파라미터 확장: `"fast"`, `"balanced"`, `"accurate"`

**변경되는 함수:**
- `process_text_schedule` - 캐싱 및 병렬 처리 추가
- `extract_info_with_gemini_json` - 캐싱 추가
- `translate_korean_to_english` - 캐싱 버전으로 래핑

#### [MODIFY] [static/index.html](file:///c:/Users/82105/PycharmProjects/Acdt_Project_Render_ver/static/index.html)

**프론트엔드 최적화:**
- 입력 필드에 디바운싱 적용 (500ms)
- 로딩 상태 개선: 프로그레스 바, 단계별 표시
- "⚡ 빠른 모드" 토글 스위치 추가 (기본값: balanced)

---

## Verification Plan

### 단계별 검증

#### 1. 클립보드 감지 테스트
1. 메모장에 "내일 오후 3시 강남역에서 친구 만나기" 복사
2. 브라우저 탭으로 돌아오기
3. ✅ 알림 배너가 뜨는지 확인
4. "분석하기" 클릭 → ✅ 자동으로 일정 추출되는지 확인

#### 2. 파일 형식 테스트
1. PDF 파일 생성 (일정 정보 포함)
2. 파일 업로드 탭에서 드래그&드롭
3. ✅ PDF 텍스트가 추출되고 일정 분석되는지 확인
4. DOCX, XLSX도 동일하게 테스트

#### 3. 템플릿 기능 테스트
1. 일정 입력 후 "템플릿으로 저장" 클릭
2. 템플릿 이름 입력 (예: "매주 팀 미팅")
3. ✅ 템플릿 드롭다운에 표시되는지 확인
4. 드롭다운에서 선택 → ✅ 자동으로 입력되는지 확인
5. 브라우저 새로고침 → ✅ 템플릿이 유지되는지 확인

#### 4. 속도 개선 테스트
1. **Before:** 현재 브랜치에서 분석 시간 측정 (5회 평균)
2. **After:** 개선 후 분석 시간 측정 (5회 평균)
3. ✅ 50% 이상 속도 향상 확인
4. ✅ spaCy 모델이 여전히 1순위로 실행되는지 로그 확인
5. "빠른 모드" vs "정확 모드" 결과 비교

### 성능 측정 기준
- **목표:** 평균 응답 시간 2.5초 이하
- **spaCy 실행:** 항상 100% (필수)
- **Gemini 호출 빈도:** 현재 대비 30% 감소
- **캐시 히트율:** 20% 이상

---

## 구현 순서

### Phase 1: 파일 형식 확장 (우선순위 1)
- 가장 실용적이고 즉시 효과를 볼 수 있음
- 백엔드만 수정하면 되어 복잡도 낮음

### Phase 2: 클립보드 감지 (우선순위 2)
- 프론트엔드만 수정
- 사용 빈도 높음

### Phase 3: 템플릿 기능 (우선순위 3)
- 프론트엔드만 수정
- localStorage 활용으로 간단함

### Phase 4: 속도 최적화 (우선순위 4)
- 기존 코드 리팩토링 필요
- 신중한 테스트 필요 (성능 유지 확인)

---

## 예상 일정

- **Phase 1 (파일 형식):** 1-2시간
- **Phase 2 (클립보드):** 1시간
- **Phase 3 (템플릿):** 1-2시간
- **Phase 4 (속도 최적화):** 2-3시간
- **통합 테스트:** 1시간

**총 예상 시간:** 6-9시간
