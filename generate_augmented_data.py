# -*- coding: utf-8 -*-
"""
NER Training Data Generator (v2)
Generates high-quality augmented training data for schedule extraction.
Focus: EVENT_TITLE, TIME ranges, LOC diversity
Used to generate data for model v2 training.
"""

import json
import random
from datetime import datetime, timedelta

# ==============================================================================
# Configuration
# ==============================================================================

OUTPUT_FILE = "augmented_training_data.json"
TARGET_COUNT = 10000  # Target number of examples to generate

# ==============================================================================
# Data Templates
# ==============================================================================

EVENT_TYPES = [
    "Meeting", "Team meeting", "1:1 meeting", "Staff meeting", "Board meeting",
    "Project meeting", "Client meeting", "Weekly sync", "Daily standup",
    "Sprint planning", "Retrospective", "Kickoff meeting", "All-hands meeting",
    "Strategy session", "Brainstorming session", "Planning session",
    "Lunch", "Dinner", "Breakfast", "Coffee", "Brunch", "Happy hour",
    "Birthday party", "Graduation party", "Housewarming", "Baby shower",
    "Wedding", "Anniversary dinner", "Date night", "Game night", "Movie night",
    "Farewell party", "Welcome party", "Celebration", "Get-together", "Potluck",
    "BBQ", "Picnic", "Beach day", "Pool party", "Karaoke night",
    "Interview", "Presentation", "Workshop", "Training", "Seminar",
    "Conference call", "Video call", "Phone call", "Demo", "Review",
    "Performance review", "Code review", "Design review", "Sales pitch",
    "Webinar", "Town hall", "Product launch", "Press conference",
    "Gym", "Yoga", "Workout", "Haircut", "Doctor appointment",
    "Dentist appointment", "Therapy session", "Massage", "Spa day",
    "Class", "Lecture", "Study session", "Exam", "Quiz", "Tutorial",
    "Office hours", "Lab session", "Group project", "Thesis defense",
    "Concert", "Movie", "Shopping", "Grocery shopping", "Hiking",
    "Swimming", "Tennis", "Golf", "Running", "Cycling", "Skiing",
    "Flight", "Train", "Bus", "Airport pickup", "Hotel check-in",
    "Hotel checkout", "Car rental", "Road trip", "Cruise departure",
]

PEOPLE = [
    "John", "Sarah", "Mike", "Emily", "David", "Jessica", "Chris", "Lisa",
    "Tom", "Amy", "Alex", "Rachel", "Daniel", "Michelle", "Kevin", "Jennifer",
    "Mom", "Dad", "parents", "family", "friends", "team", "clients", "boss",
]

LOCATIONS = [
    "Starbucks", "Blue Bottle", "the cafe", "the coffee shop",
    "the restaurant", "Italian restaurant", "Korean BBQ", "sushi place",
    "the office", "Conference Room A", "Meeting Room 3", "the boardroom",
    "the library", "the park", "Central Park", "the mall", "downtown",
    "Gangnam", "Hongdae", "Itaewon", "Myeongdong", "Seoul Station",
    "home", "my place", "your place", "the gym", "the hotel",
    "the campus", "Student Center", "the lecture hall", "Room 101",
]

TIMES = [
    "9am", "10am", "11am", "12pm", "1pm", "2pm", "3pm", "4pm", "5pm",
    "6pm", "7pm", "8pm", "9pm", "10pm",
    "9:00 AM", "10:30 AM", "11:15 AM", "12:00 PM", "1:30 PM", "2:45 PM",
    "3:00 PM", "4:30 PM", "5:15 PM", "6:00 PM", "7:30 PM", "8:00 PM",
    "noon", "midnight", "morning", "afternoon", "evening",
]

END_TIMES = [
    "11am", "12pm", "1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm",
    "11:00 AM", "12:00 PM", "1:30 PM", "2:00 PM", "3:30 PM", "5:00 PM",
]

DATES = [
    "today", "tomorrow", "the day after tomorrow",
    "this Monday", "this Tuesday", "this Wednesday", "this Thursday",
    "this Friday", "this Saturday", "this Sunday",
    "next Monday", "next Tuesday", "next Wednesday", "next Thursday",
    "next Friday", "next Saturday", "next Sunday",
    "January 5th", "February 14th", "March 20th", "April 1st",
    "May 15th", "June 30th", "July 4th", "December 25th",
]

END_DATES = ["the 20th", "the 25th", "the 30th", "next Friday", "next month"]

TEMPLATES = [
    "{event} at {time}.", "{event} {date}.", "{event} at {location}.",
    "{event} with {person}.", "{event} {date} at {time}.",
    "{event} at {location} {date}.", "{event} with {person} at {time}.",
    "{event} with {person} at {location} {date} at {time}.",
    "Meet me at {location} at {time}.", "Let's meet at {location} {date}.",
    "I have {event} at {time}.", "I have {event} {date}.",
    "We have {event} {date} at {time}.",
    "{event} from {time} to {end_time}.",
    "{event} from {time} to {end_time} at {location}.",
    "{event} starts at {time} and ends at {end_time}.",
    "{event} from {date} to {end_date}.",
    "Schedule {event} for {date} at {time}.",
    "Book {event} at {location} {date}.",
]

def generate_single_example():
    template = random.choice(TEMPLATES)
    event = random.choice(EVENT_TYPES) if "{event}" in template else ""
    person = random.choice(PEOPLE) if "{person}" in template else ""
    location = random.choice(LOCATIONS) if "{location}" in template else ""
    time_val = random.choice(TIMES) if "{time}" in template else ""
    end_time = random.choice(END_TIMES) if "{end_time}" in template else ""
    date_val = random.choice(DATES) if "{date}" in template else ""
    end_date = random.choice(END_DATES) if "{end_date}" in template else ""
    
    text = template.format(event=event, person=person, location=location,
                          time=time_val, end_time=end_time,
                          date=date_val, end_date=end_date)
    
    return {
        "Text": text, "Date_Entity": date_val, "Time_Entity": time_val,
        "End_Date_Entity": end_date, "End_Time_Entity": end_time,
        "Location_Entity": location, "Event_Entity": event, "Notes": "augmented"
    }

def generate_event_focused():
    event = random.choice(EVENT_TYPES)
    patterns = [f"{event}.", f"I have {event}.", f"Don't forget {event}.",
                f"Reminder: {event}.", f"Quick {event}.", f"The {event}."]
    return {"Text": random.choice(patterns), "Date_Entity": "", "Time_Entity": "",
            "End_Date_Entity": "", "End_Time_Entity": "", "Location_Entity": "",
            "Event_Entity": event, "Notes": "event_focused"}

def generate_time_range():
    event = random.choice(EVENT_TYPES)
    time_val = random.choice(TIMES[:14])
    end_time = random.choice(END_TIMES)
    location = random.choice(LOCATIONS) if random.random() > 0.5 else ""
    patterns = [f"{event} from {time_val} to {end_time}.",
                f"The {event} runs from {time_val} to {end_time}.",
                f"{event}: {time_val} - {end_time}."]
    return {"Text": random.choice(patterns), "Date_Entity": "", "Time_Entity": time_val,
            "End_Date_Entity": "", "End_Time_Entity": end_time,
            "Location_Entity": location, "Event_Entity": event, "Notes": "time_range"}

def generate_location_focused():
    location = random.choice(LOCATIONS)
    event = random.choice(EVENT_TYPES) if random.random() > 0.3 else ""
    patterns = [f"Meet me at {location}.", f"See you at {location}.",
                f"Let's meet at {location}.", f"I'll be at {location}."]
    if event: patterns.append(f"{event} at {location}.")
    return {"Text": random.choice(patterns), "Date_Entity": "", "Time_Entity": "",
            "End_Date_Entity": "", "End_Time_Entity": "",
            "Location_Entity": location, "Event_Entity": event, "Notes": "location_focused"}

def generate_all_data(target_count=TARGET_COUNT):
    data = []
    counts = [int(target_count*0.4), int(target_count*0.3), int(target_count*0.15), int(target_count*0.15)]
    for _ in range(counts[0]): data.append(generate_single_example())
    for _ in range(counts[1]): data.append(generate_event_focused())
    for _ in range(counts[2]): data.append(generate_time_range())
    for _ in range(counts[3]): data.append(generate_location_focused())
    random.shuffle(data)
    seen = set()
    unique = [d for d in data if d["Text"] not in seen and not seen.add(d["Text"])]
    print(f"Generated {len(unique)} unique examples")
    return unique

if __name__ == "__main__":
    print("=" * 60)
    print("NER Training Data Generator (v2)")
    print("=" * 60)
    data = generate_all_data(TARGET_COUNT)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)
