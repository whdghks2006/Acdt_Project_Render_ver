# -*- coding: utf-8 -*-
"""
벤치마크 스크립트: 일정 추출 성능 측정
Before/After 비교를 위한 테스트 및 측정 도구
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 서버 URL (로컬 테스트용)
BASE_URL = "http://localhost:7860"

# ============================================================================
# 테스트 케이스 정의 (10개)
# ============================================================================
def get_test_cases() -> List[Dict[str, Any]]:
    """테스트 케이스 반환 - 정답(expected)과 함께"""
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    # 다음주 금요일 계산
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday + 7)
    
    return [
        {
            "id": 1,
            "text": "내일 오후 3시에 강남역에서 미팅",
            "expected": {
                "start_date": tomorrow.strftime("%Y-%m-%d"),
                "start_time": "15:00",
                "location": "강남역"
            },
            "lang": "ko"
        },
        {
            "id": 2,
            "text": "12월 25일 크리스마스 파티",
            "expected": {
                "start_date": "2025-12-25",
                "start_time": "",
                "location": ""
            },
            "lang": "ko"
        },
        {
            "id": 3,
            "text": "다음주 금요일 오전 10시 회의",
            "expected": {
                "start_date": next_friday.strftime("%Y-%m-%d"),
                "start_time": "10:00",
                "location": ""
            },
            "lang": "ko"
        },
        {
            "id": 4,
            "text": "2025년 1월 15일 신년회",
            "expected": {
                "start_date": "2025-01-15",
                "start_time": "",
                "location": ""
            },
            "lang": "ko"
        },
        {
            "id": 5,
            "text": "매주 월요일 9시 정례회의",
            "expected": {
                "start_date": "",
                "start_time": "09:00",
                "location": ""
            },
            "lang": "ko"
        },
        {
            "id": 6,
            "text": "Meeting at 3pm tomorrow",
            "expected": {
                "start_date": tomorrow.strftime("%Y-%m-%d"),
                "start_time": "15:00",
                "location": ""
            },
            "lang": "en"
        },
        {
            "id": 7,
            "text": "Dec 25 Christmas dinner at home",
            "expected": {
                "start_date": "2025-12-25",
                "start_time": "",
                "location": "home"
            },
            "lang": "en"
        },
        {
            "id": 8,
            "text": "next Friday 2:30 PM conference call",
            "expected": {
                "start_date": next_friday.strftime("%Y-%m-%d"),
                "start_time": "14:30",
                "location": ""
            },
            "lang": "en"
        },
        {
            "id": 9,
            "text": "오후 2시부터 4시까지 세미나",
            "expected": {
                "start_date": "",
                "start_time": "14:00",
                "location": ""
            },
            "lang": "ko"
        },
        {
            "id": 10,
            "text": "12/20 at Seoul Station",
            "expected": {
                "start_date": "2025-12-20",
                "start_time": "",
                "location": "Seoul Station"
            },
            "lang": "en"
        }
    ]


# ============================================================================
# 측정 함수들
# ============================================================================
def test_single_case(case: Dict) -> Dict[str, Any]:
    """단일 테스트 케이스 실행 및 결과 반환"""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/extract",
            json={"text": case["text"], "lang": case["lang"], "mode": "full"},
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                "id": case["id"],
                "text": case["text"],
                "expected": case["expected"],
                "actual": {
                    "start_date": result.get("start_date", ""),
                    "start_time": result.get("start_time", ""),
                    "location": result.get("location", "")
                },
                "used_model": result.get("used_model", ""),
                "response_time": elapsed,
                "success": True,
                "gemini_called": "Gemini" in result.get("used_model", "") or "Smart" in result.get("used_model", "")
            }
        else:
            return {
                "id": case["id"],
                "text": case["text"],
                "error": f"HTTP {response.status_code}",
                "response_time": elapsed,
                "success": False
            }
    except Exception as e:
        return {
            "id": case["id"],
            "text": case["text"],
            "error": str(e),
            "response_time": time.time() - start_time,
            "success": False
        }


def calculate_accuracy(results: List[Dict]) -> Dict[str, float]:
    """정확도 계산"""
    date_correct = 0
    date_total = 0
    time_correct = 0
    time_total = 0
    loc_correct = 0
    loc_total = 0
    
    for r in results:
        if not r.get("success"):
            continue
        
        exp = r["expected"]
        act = r["actual"]
        
        # DATE 정확도
        if exp["start_date"]:
            date_total += 1
            if exp["start_date"] == act["start_date"]:
                date_correct += 1
        
        # TIME 정확도
        if exp["start_time"]:
            time_total += 1
            if exp["start_time"] == act["start_time"]:
                time_correct += 1
        
        # LOCATION 정확도
        if exp["location"]:
            loc_total += 1
            # 부분 매칭 허용 (대소문자 무시)
            if exp["location"].lower() in act["location"].lower() or \
               act["location"].lower() in exp["location"].lower():
                loc_correct += 1
    
    return {
        "date_accuracy": (date_correct / date_total * 100) if date_total > 0 else 0,
        "time_accuracy": (time_correct / time_total * 100) if time_total > 0 else 0,
        "location_accuracy": (loc_correct / loc_total * 100) if loc_total > 0 else 0,
        "date_stats": f"{date_correct}/{date_total}",
        "time_stats": f"{time_correct}/{time_total}",
        "location_stats": f"{loc_correct}/{loc_total}"
    }


def run_benchmark(mode: str = "before") -> Dict[str, Any]:
    """전체 벤치마크 실행"""
    print(f"\n{'='*60}")
    print(f"🔬 벤치마크 실행: {mode.upper()}")
    print(f"{'='*60}")
    
    test_cases = get_test_cases()
    results = []
    
    for case in test_cases:
        print(f"\n[{case['id']}/10] 테스트 중: {case['text'][:30]}...")
        result = test_single_case(case)
        results.append(result)
        
        if result["success"]:
            print(f"  ✅ 응답시간: {result['response_time']:.2f}s")
            print(f"  📊 사용 모델: {result['used_model']}")
            print(f"  📅 Date: {result['actual']['start_date']} (기대: {result['expected']['start_date']})")
            print(f"  ⏰ Time: {result['actual']['start_time']} (기대: {result['expected']['start_time']})")
            print(f"  📍 Loc: {result['actual']['location']} (기대: {result['expected']['location']})")
        else:
            print(f"  ❌ 오류: {result.get('error')}")
    
    # 통계 계산
    successful = [r for r in results if r.get("success")]
    accuracy = calculate_accuracy(results)
    
    gemini_calls = sum(1 for r in successful if r.get("gemini_called"))
    avg_response_time = sum(r["response_time"] for r in successful) / len(successful) if successful else 0
    
    summary = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(test_cases),
        "successful_tests": len(successful),
        "accuracy": accuracy,
        "gemini_call_rate": (gemini_calls / len(successful) * 100) if successful else 0,
        "gemini_calls": gemini_calls,
        "avg_response_time": avg_response_time,
        "results": results
    }
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"📊 결과 요약 ({mode.upper()})")
    print(f"{'='*60}")
    print(f"✅ 성공: {len(successful)}/{len(test_cases)}")
    print(f"📅 DATE 정확도: {accuracy['date_accuracy']:.1f}% ({accuracy['date_stats']})")
    print(f"⏰ TIME 정확도: {accuracy['time_accuracy']:.1f}% ({accuracy['time_stats']})")
    print(f"📍 LOC 정확도: {accuracy['location_accuracy']:.1f}% ({accuracy['location_stats']})")
    print(f"🤖 Gemini 호출률: {summary['gemini_call_rate']:.1f}% ({gemini_calls}/{len(successful)})")
    print(f"⚡ 평균 응답시간: {avg_response_time:.2f}초")
    
    # 파일로 저장
    filename = f"benchmark_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과 저장: {filename}")
    
    return summary


def compare_results(before_file: str, after_file: str):
    """Before/After 결과 비교"""
    with open(before_file, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(after_file, "r", encoding="utf-8") as f:
        after = json.load(f)
    
    print(f"\n{'='*60}")
    print("📊 Before vs After 비교")
    print(f"{'='*60}")
    
    headers = ["지표", "Before", "After", "변화"]
    rows = [
        ("DATE 정확도", 
         f"{before['accuracy']['date_accuracy']:.1f}%",
         f"{after['accuracy']['date_accuracy']:.1f}%",
         f"{after['accuracy']['date_accuracy'] - before['accuracy']['date_accuracy']:+.1f}%"),
        ("TIME 정확도",
         f"{before['accuracy']['time_accuracy']:.1f}%",
         f"{after['accuracy']['time_accuracy']:.1f}%",
         f"{after['accuracy']['time_accuracy'] - before['accuracy']['time_accuracy']:+.1f}%"),
        ("LOC 정확도",
         f"{before['accuracy']['location_accuracy']:.1f}%",
         f"{after['accuracy']['location_accuracy']:.1f}%",
         f"{after['accuracy']['location_accuracy'] - before['accuracy']['location_accuracy']:+.1f}%"),
        ("Gemini 호출률",
         f"{before['gemini_call_rate']:.1f}%",
         f"{after['gemini_call_rate']:.1f}%",
         f"{after['gemini_call_rate'] - before['gemini_call_rate']:+.1f}%"),
        ("평균 응답시간",
         f"{before['avg_response_time']:.2f}s",
         f"{after['avg_response_time']:.2f}s",
         f"{after['avg_response_time'] - before['avg_response_time']:+.2f}s"),
    ]
    
    # 출력
    col_widths = [15, 12, 12, 12]
    print(f"| {' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))} |")
    print(f"|{'-'*15}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for row in rows:
        print(f"| {' | '.join(str(c).ljust(w) for c, w in zip(row, col_widths))} |")


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python benchmark.py before    # Before 측정")
        print("  python benchmark.py after     # After 측정")
        print("  python benchmark.py compare before.json after.json  # 비교")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "before":
        run_benchmark("before")
    elif mode == "after":
        run_benchmark("after")
    elif mode == "compare" and len(sys.argv) >= 4:
        compare_results(sys.argv[2], sys.argv[3])
    else:
        print("잘못된 명령어입니다.")
