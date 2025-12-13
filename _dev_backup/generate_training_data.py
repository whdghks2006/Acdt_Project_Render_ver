# -*- coding: utf-8 -*-
"""
합성 데이터 생성기 (END 라벨 포함)
✅ START+END 있는 문장 위주로 학습할 수 있도록 수정된 버전
synthetic_training_data.json 형식으로 출력
"""

import json
import random
import os

# ==========================================
# A. 단어 데이터베이스 (DB)
# ==========================================

locations = [
    # 일반 장소
    "Gangnam Station", "Seoul Station", "Times Square", "Building 301", "Room 101",
    "the cafeteria", "Starbucks", "the library", "Central Park", "the lobby",
    "Conference Room A", "the gym", "my office", "New York", "London",
    "the meeting room", "Zoom", "Google Meet", "Teams", "the office",
    "Room 205", "the auditorium", "Main Hall", "Building A", "the rooftop",
    # 대학/학교
    "Room 302", "Lecture Hall B", "the student center", "Science Building", "the campus",
    "Engineering Hall", "the dormitory", "the lab", "Computer Science Building", "Library Floor 3",
    # 회사/비즈니스
    "the boardroom", "HR Office", "Floor 15", "the reception area", "CEO's office",
    "Marketing Department", "IT Room", "the break room", "Training Room 1", "the warehouse",
    # 레스토랑/카페
    "Blue Bottle", "the pizza place", "Italian restaurant", "McDonald's", "the food court",
    "a nearby cafe", "Subway", "the sushi place", "the Korean BBQ", "the buffet",
    # 공공장소
    "City Hall", "the hospital", "the airport", "the train station", "the bus stop",
    "the post office", "the bank", "the mall", "the museum", "the park",
    # 온라인
    "Microsoft Teams", "Slack huddle", "Discord", "Webex", "online",
    "virtually", "remotely", "via video call", "on a call", "the virtual meeting room"
]

events = [
    # 업무 관련
    "Team meeting", "Project deadline", "Client meeting", "Sprint planning", "Code review",
    "Performance review", "Budget meeting", "Strategy session", "Quarterly review", "Board meeting",
    "One-on-one", "Standup meeting", "Retrospective", "Product demo", "Sales pitch",
    "Onboarding session", "Exit interview", "Team building", "Brainstorming session", "Workshop",
    # 개인 일정
    "Doctor's appointment", "Dentist appointment", "Haircut", "Car service", "Grocery shopping",
    "Gym session", "Yoga class", "Therapy session", "Massage appointment", "Eye exam",
    # 사교/파티
    "Birthday party", "Wedding ceremony", "Dinner party", "Graduation party", "Baby shower",
    "Housewarming", "Farewell party", "Anniversary dinner", "Reunion", "Potluck",
    "Lunch with Sarah", "Coffee with Mike", "Drinks with colleagues", "Brunch meetup", "Game night",
    # 학교/교육
    "Final exam", "Midterm exam", "Lecture", "Tutorial", "Lab session",
    "Thesis defense", "Study group", "Office hours", "Orientation", "Seminar",
    "Research presentation", "Guest lecture", "Club meeting", "Career fair", "Internship interview",
    # 이벤트/행사
    "Conference", "Hackathon", "Concert", "Festival", "Exhibition",
    "Trade show", "Networking event", "Award ceremony", "Charity gala", "Webinar",
    # 스포츠/취미
    "Soccer match", "Basketball game", "Tennis lesson", "Golf outing", "Swimming practice",
    "Running club", "Cycling tour", "Hiking trip", "Dance class", "Cooking class"
]

# 시작 날짜 (다양한 표현)
start_dates = [
    # 상대적 표현
    "tomorrow", "today", "the day after tomorrow", "next Monday", "next Tuesday",
    "next Wednesday", "next Thursday", "next Friday", "next Saturday", "next Sunday",
    "this Monday", "this Tuesday", "this Wednesday", "this Thursday", "this Friday",
    "this Saturday", "this Sunday", "this week", "next week", "this weekend",
    "next weekend", "in two days", "in three days", "in a week", "in two weeks",
    # 절대적 표현 (월/일)
    "January 15th", "January 20th", "February 1st", "February 14th", "March 1st",
    "March 10th", "March 15th", "April 1st", "April 15th", "May 1st",
    "May 20th", "June 1st", "June 15th", "July 4th", "July 20th",
    "August 1st", "August 15th", "September 1st", "September 20th", "October 1st",
    "October 31st", "November 5th", "November 15th", "December 1st", "December 25th",
    # 요일/기타
    "on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday",
    "the 1st", "the 5th", "the 10th", "the 15th", "the 20th", "the 25th", "the last day of the month"
]

# 종료 날짜 (기간 표현용)
end_dates = [
    "next Friday", "the 20th", "December 31st", "the end of the month",
    "Sunday", "next week Friday", "the following Monday", "March 15th",
    "the 25th", "next Thursday", "April 5th", "the weekend",
    "the end of the week", "next Sunday", "the following Friday", "February 28th",
    "June 30th", "December 15th", "the end of the year", "next month",
    "in two weeks", "the 30th", "the last day", "next Wednesday"
]

# 시작 시간 (다양한 형식)
start_times = [
    # 12시간 형식
    "2 PM", "10 AM", "7 PM", "8 AM", "9 AM", "11 AM", "1 PM", "3 PM", "4 PM", "5 PM",
    "6 PM", "7 AM", "12 PM", "6 AM", "9 PM", "10 PM", "8 PM",
    # 분 포함
    "9:30 AM", "10:30 AM", "11:30 AM", "12:30 PM", "1:30 PM", "2:30 PM",
    "3:30 PM", "4:30 PM", "5:30 PM", "6:30 PM", "7:30 PM", "8:30 PM",
    "9:15 AM", "10:15 AM", "2:15 PM", "3:15 PM", "4:15 PM", "5:15 PM",
    "9:45 AM", "10:45 AM", "2:45 PM", "3:45 PM", "4:45 PM",
    # 24시간 형식
    "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00",
    "09:00", "10:00", "11:00", "12:00", "13:00",
    # 자연어 표현
    "noon", "midnight", "morning", "afternoon", "evening",
    "in the morning", "in the afternoon", "in the evening", "at dawn", "at dusk"
]

# 종료 시간
end_times = [
    # 12시간 형식
    "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM",
    "12 PM", "1 PM", "2 PM", "3 PM", "11 AM", "10 AM",
    # 분 포함
    "4:30 PM", "5:30 PM", "6:30 PM", "7:30 PM", "8:30 PM", "9:30 PM",
    "3:30 PM", "2:30 PM", "1:30 PM", "12:30 PM", "11:30 AM", "10:30 AM",
    "5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM",
    # 24시간 형식
    "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",
    "14:00", "15:00", "16:00",
    # 자연어 표현
    "noon", "midnight", "end of day", "close of business", "late evening"
]

# ==========================================
# B. 문장 템플릿 (START + END 포함)
# ==========================================

# START만 있는 템플릿
# 👉 이번 버전에서는 새로 생성하지 않음 (이미 충분히 있음)
start_only_templates = [
    {"template": "I have a {event} at {loc} {start_date} at {start_time}.",
     "has": ["event", "loc", "start_date", "start_time"]},
    {"template": "{start_date}, there is a {event} at {start_time}.",
     "has": ["start_date", "event", "start_time"]},
    {"template": "Let's meet at {loc} at {start_time} for {event}.",
     "has": ["loc", "start_time", "event"]},
    {"template": "Reminder: {event} is {start_date} at {start_time}.",
     "has": ["event", "start_date", "start_time"]},
    {"template": "Please attend the {event} at {loc}.",
     "has": ["event", "loc"]},
    {"template": "My {event} starts at {start_time} {start_date}.",
     "has": ["event", "start_time", "start_date"]},
    {"template": "Go to {loc} {start_date} for the {event}.",
     "has": ["loc", "start_date", "event"]},
    {"template": "{event} will be held at {loc} on {start_date}.",
     "has": ["event", "loc", "start_date"]},
    {"template": "Schedule a {event} at {start_time}.",
     "has": ["event", "start_time"]},
    {"template": "{start_date} is the deadline for {event}.",
     "has": ["start_date", "event"]},
]

# START + END 포함 템플릿 (핵심!)
start_end_templates = [
    {"template": "Meeting from {start_time} to {end_time} {start_date} at {loc}.",
     "has": ["start_time", "end_time", "start_date", "loc"]},
    {"template": "The {event} runs from {start_date} to {end_date}.",
     "has": ["event", "start_date", "end_date"]},
    {"template": "{event} at {loc} from {start_time} until {end_time}.",
     "has": ["event", "loc", "start_time", "end_time"]},
    {"template": "Workshop from {start_time} to {end_time} on {start_date}.",
     "has": ["start_time", "end_time", "start_date"]},
    {"template": "The conference is scheduled from {start_date} through {end_date}.",
     "has": ["start_date", "end_date"]},
    {"template": "{event} starts at {start_time} and ends at {end_time}.",
     "has": ["event", "start_time", "end_time"]},
    {"template": "Book {loc} from {start_time} to {end_time} for {event}.",
     "has": ["loc", "start_time", "end_time", "event"]},
    {"template": "Training session from {start_date} to {end_date} at {loc}.",
     "has": ["start_date", "end_date", "loc"]},
    {"template": "The {event} is from {start_time} to {end_time} at {loc}.",
     "has": ["event", "start_time", "end_time", "loc"]},
    {"template": "Please block {start_date} to {end_date} for the {event}.",
     "has": ["start_date", "end_date", "event"]},
    {"template": "Join us from {start_time} until {end_time} for the {event}.",
     "has": ["start_time", "end_time", "event"]},
    {"template": "Available from {start_time} to {end_time} on {start_date}.",
     "has": ["start_time", "end_time", "start_date"]},
]

# ==========================================
# C. 합성 데이터 생성 함수
# ==========================================

### ✅ 변경 포인트 1:
###    START-only는 새로 만들지 않고,
###    START+END 샘플만 집중적으로 생성하도록 수정
def generate_synthetic_data(num_start_end=500):
    """합성 데이터 생성 (START+END 위주)"""
    dataset = []
    current_id = 10001  # 기존 데이터와 ID 충돌 방지
    
    print("🔄 합성 데이터 생성 중... (START+END 중심)")
    
    # START + END 데이터 생성
    for _ in range(num_start_end):
        template_info = random.choice(start_end_templates)
        template = template_info["template"]
        has_fields = template_info["has"]
        
        # 랜덤 값 선택
        values = {
            "event": random.choice(events),
            "loc": random.choice(locations),
            "start_date": random.choice(start_dates),
            "end_date": random.choice(end_dates),
            "start_time": random.choice(start_times),
            "end_time": random.choice(end_times),
        }
        
        # 템플릿에 값 채우기
        text = template.format(**values)
        
        # 데이터 레코드 생성
        record = {
            "ID": current_id,
            "Text": text,
            "Date_Entity": values["start_date"] if "start_date" in has_fields else "",
            "Time_Entity": values["start_time"] if "start_time" in has_fields else "",
            "Location_Entity": values["loc"] if "loc" in has_fields else "",
            "Event_Entity": values["event"] if "event" in has_fields else "",
            "End_Date_Entity": values["end_date"] if "end_date" in has_fields else "",
            "End_Time_Entity": values["end_time"] if "end_time" in has_fields else "",
            "Notes": "synthetic_with_end"
        }
        
        dataset.append(record)
        current_id += 1
    
    print(f"  ✅ START+END 데이터 {num_start_end}개 생성")
    
    return dataset


# ==========================================
# D. 메인 실행
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("📊 NER 학습용 합성 데이터 생성기 (START+END 중심)")
    print("=" * 50)
    
    # 1. 합성 데이터 생성
    ### ✅ 변경 포인트 2:
    ###    더 이상 START-only 개수 인자 사용 안 함
    synthetic_data = generate_synthetic_data(num_start_end=2500)
    print(f"\n📈 총 합성 데이터: {len(synthetic_data)}개 (모두 START+END 형태)")
    
    # 2. 기존 데이터 로드 및 병합 (선택적)
    existing_file = 'Data Set v1_G14-1.json'
    final_dataset = synthetic_data.copy()
    
    if os.path.exists(existing_file):
        try:
            with open(existing_file, 'r', encoding='utf-8') as f:
                real_data = json.load(f)
            
            # 기존 데이터에 End 필드가 없으면 추가
            for record in real_data:
                if "End_Date_Entity" not in record:
                    record["End_Date_Entity"] = ""
                if "End_Time_Entity" not in record:
                    record["End_Time_Entity"] = ""
            
            print(f"\n📂 기존 파일 '{existing_file}' 로드 성공! ({len(real_data)}개)")
            
            # 데이터 병합
            final_dataset = synthetic_data + real_data
            print(f"🔗 데이터 병합 완료: 합성({len(synthetic_data)}) + 실제({len(real_data)}) = 총 {len(final_dataset)}개")
            
        except Exception as e:
            print(f"⚠️ 기존 파일 로드 중 오류 발생: {e}")
            print("👉 합성 데이터만 사용합니다.")
    else:
        print(f"\n⚠️ '{existing_file}' 파일이 없습니다. 합성 데이터만 사용합니다.")
    
    # 3. 데이터 섞기
    random.shuffle(final_dataset)
    
    # 4. 저장
    output_file = 'synthetic_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 최종 데이터셋 저장 완료!")
    print(f"   📁 파일명: {output_file}")
    print(f"   📊 총 데이터 수: {len(final_dataset)}개")
    
    # 5. 예시 출력
    print("\n📋 예시 데이터:")
    for i, example in enumerate(final_dataset[:3]):
        print(f"\n--- 예시 {i+1} ---")
        print(f"  Text: {example['Text']}")
        print(f"  Date: {example['Date_Entity']}")
        print(f"  Time: {example['Time_Entity']}")
        print(f"  End Date: {example.get('End_Date_Entity', '')}")
        print(f"  End Time: {example.get('End_Time_Entity', '')}")
        print(f"  Location: {example['Location_Entity']}")
        print(f"  Event: {example['Event_Entity']}")
    
    print("\n" + "=" * 50)
