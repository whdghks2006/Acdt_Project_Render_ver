"""
AI Smart Scheduler - Performance Benchmark Script
최적화 전후 성능 비교를 위한 벤치마크 도구
"""

import requests
import time
import statistics
import json

BASE_URL = "http://localhost:7860"

# 테스트 케이스
TEST_CASES = [
    {
        "name": "단일 일정 (영어)",
        "text": "Meeting tomorrow at 3pm at Starbucks",
        "expected_type": "single"
    },
    {
        "name": "단일 일정 (한글)",
        "text": "내일 오후 3시에 강남역 스타벅스에서 미팅",
        "expected_type": "single"
    },
    {
        "name": "다중 일정 (영어)",
        "text": "Next Friday I have a doctor's appointment at 10am at Seoul Hospital, and afterwards I'm meeting my friend for lunch at 12:30pm at Myeongdong. Also, at 6pm there's a team dinner at Gangnam.",
        "expected_type": "multiple"
    },
    {
        "name": "다중 일정 (한글)",
        "text": "13일 오후 2시에 병원 예약 있고, 14일 저녁 7시에 친구들이랑 홍대에서 저녁",
        "expected_type": "multiple"
    },
]

def run_benchmark(iterations: int = 3):
    """벤치마크 실행"""
    print("=" * 60)
    print("🚀 AI Smart Scheduler - Performance Benchmark")
    print("=" * 60)
    print(f"서버: {BASE_URL}")
    print(f"반복 횟수: {iterations}회")
    print()
    
    results = []
    
    for case in TEST_CASES:
        print(f"📌 테스트: {case['name']}")
        print(f"   입력: {case['text'][:50]}...")
        
        times = []
        
        for i in range(iterations):
            start = time.time()
            
            try:
                response = requests.post(
                    f"{BASE_URL}/extract",
                    json={"text": case["text"], "lang": "en", "mode": "full"},
                    timeout=60
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    times.append(elapsed)
                    data = response.json()
                    schedule_count = len(data.get("schedules", [])) or 1
                    print(f"   [{i+1}/{iterations}] ✅ {elapsed:.3f}초 (일정 {schedule_count}개)")
                else:
                    print(f"   [{i+1}/{iterations}] ❌ HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"   [{i+1}/{iterations}] ⏱️ 타임아웃 (60초)")
            except Exception as e:
                print(f"   [{i+1}/{iterations}] ❌ 오류: {e}")
        
        if times:
            avg = statistics.mean(times)
            min_t = min(times)
            max_t = max(times)
            
            results.append({
                "name": case["name"],
                "avg": avg,
                "min": min_t,
                "max": max_t,
                "count": len(times)
            })
            
            print(f"   📊 평균: {avg:.3f}초 | 최소: {min_t:.3f}초 | 최대: {max_t:.3f}초")
        print()
    
    # 요약
    print("=" * 60)
    print("📊 벤치마크 결과 요약")
    print("=" * 60)
    print(f"{'테스트 케이스':<25} {'평균(초)':<12} {'최소':<12} {'최대':<12}")
    print("-" * 60)
    
    total_avg = 0
    for r in results:
        print(f"{r['name']:<25} {r['avg']:<12.3f} {r['min']:<12.3f} {r['max']:<12.3f}")
        total_avg += r['avg']
    
    print("-" * 60)
    print(f"{'전체 평균':<25} {total_avg/len(results) if results else 0:<12.3f}")
    print()
    
    # JSON 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "iterations": iterations,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 결과 저장: {filename}")
    
    return results


if __name__ == "__main__":
    import sys
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_benchmark(iterations)
