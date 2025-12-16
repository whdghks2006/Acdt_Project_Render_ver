# 🤖 NER 학습 데이터 생성용 AI 프롬프트

다양한 AI (ChatGPT, Claude, Gemini 등)에게 학습 데이터 생성을 요청할 때 사용하세요.

---

## 📋 프롬프트 (복사해서 사용)

```
당신은 일정 관련 NER(Named Entity Recognition) 학습 데이터를 생성하는 전문가입니다.

다음 JSON 형식으로 100개의 다양한 일정 관련 문장을 생성해주세요:

{
  "ID": 1,
  "Text": "문장 내용",
  "Date_Entity": "시작 날짜 (문장에서 추출된 그대로)",
  "Time_Entity": "시작 시간 (문장에서 추출된 그대로)",
  "Location_Entity": "장소 (문장에서 추출된 그대로)",
  "Event_Entity": "이벤트/일정 이름 (문장에서 추출된 그대로)",
  "End_Date_Entity": "종료 날짜 (문장에서 추출된 그대로, 없으면 빈 문자열)",
  "End_Time_Entity": "종료 시간 (문장에서 추출된 그대로, 없으면 빈 문자열)",
  "Notes": ""
}

### 중요 규칙:
1. **Entity 값은 반드시 Text에 있는 그대로** 추출하세요 (변환하지 마세요)
2. **다양한 문장 스타일**: 공식 이메일, 캐주얼 대화, 공지사항, 메모 등
3. **다양한 시간 표현**: "tomorrow", "next Monday", "at 3 PM", "2025-01-15", "이번 주 금요일" 등
4. **기간 표현 필수 포함** (최소 30개): "from 2pm to 4pm", "until 6 PM", "through Friday" 등
5. **영어 문장**으로 작성하세요
6. 해당 Entity가 문장에 없으면 **빈 문자열("")**로 남기세요

### 예시:

[
  {
    "ID": 1,
    "Text": "Team meeting at Conference Room B tomorrow from 2 PM to 4 PM.",
    "Date_Entity": "tomorrow",
    "Time_Entity": "2 PM",
    "Location_Entity": "Conference Room B",
    "Event_Entity": "Team meeting",
    "End_Date_Entity": "",
    "End_Time_Entity": "4 PM",
    "Notes": ""
  },
  {
    "ID": 2,
    "Text": "The workshop runs from January 15th through January 20th at the Innovation Center.",
    "Date_Entity": "January 15th",
    "Time_Entity": "",
    "Location_Entity": "the Innovation Center",
    "Event_Entity": "workshop",
    "End_Date_Entity": "January 20th",
    "End_Time_Entity": "",
    "Notes": ""
  },
  {
    "ID": 3,
    "Text": "Don't forget about the birthday party at Sarah's house this Saturday at 7 PM.",
    "Date_Entity": "this Saturday",
    "Time_Entity": "7 PM",
    "Location_Entity": "Sarah's house",
    "Event_Entity": "birthday party",
    "End_Date_Entity": "",
    "End_Time_Entity": "",
    "Notes": ""
  },
  {
    "ID": 4,
    "Text": "Available for calls between 10 AM and noon.",
    "Date_Entity": "",
    "Time_Entity": "10 AM",
    "Location_Entity": "",
    "Event_Entity": "",
    "End_Date_Entity": "",
    "End_Time_Entity": "noon",
    "Notes": ""
  }
]

### 포함해야 할 문장 유형:
1. **공식 이메일/공지** (30개): "Dear team, We are pleased to announce..."
2. **캐주얼 대화** (30개): "Hey, let's meet at...", "How about tomorrow at..."
3. **기간이 있는 일정** (30개): "from...to...", "until...", "through..."
4. **짧은 메모** (10개): "Dentist at 3", "Flight 2pm"

JSON 배열 형태로 100개를 생성해주세요.
```

---

## 💡 사용 팁

1. **여러 AI에게 각각 요청**: ChatGPT, Claude, Gemini 등에게 각각 100개씩 요청
2. **ID는 나중에 수정**: 각 AI에서 받은 데이터의 ID를 나중에 통합할 때 다시 부여
3. **검수 필수**: AI가 가끔 Entity를 잘못 추출할 수 있으므로 검수 필요
4. **병합 방법**: 모든 JSON을 하나의 배열로 합친 후 `synthetic_training_data.json`에 추가

---

## 🔄 데이터 병합 스크립트

받은 데이터를 기존 데이터에 병합할 때:

```python
import json

# 기존 데이터 로드
with open('synthetic_training_data.json', 'r', encoding='utf-8') as f:
    existing_data = json.load(f)

# 새 AI 생성 데이터 로드 (여러 파일)
new_data = []
for filename in ['chatgpt_data.json', 'claude_data.json', 'gemini_data.json']:
    with open(filename, 'r', encoding='utf-8') as f:
        new_data.extend(json.load(f))

# ID 재부여
max_id = max(item['ID'] for item in existing_data)
for i, item in enumerate(new_data):
    item['ID'] = max_id + i + 1

# 병합 및 저장
final_data = existing_data + new_data
with open('synthetic_training_data_v2.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"총 {len(final_data)}개 데이터 병합 완료!")
```
