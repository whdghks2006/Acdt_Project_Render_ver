# -*- coding: utf-8 -*-
"""
Pattern Matcher for Schedule Extraction
한국어/영어 일정 정보 추출을 위한 커스텀 패턴 매처
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


class SchedulePatternMatcher:
    """정규식 기반 일정 정보 추출기"""
    
    def __init__(self):
        # 한국어 날짜 패턴 (더 유연하게)
        self.korean_date_patterns = [
            # 절대 날짜
            (r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', 'absolute_full'),
            (r'(\d{1,2})월\s*(\d{1,2})일', 'absolute_short'),
            (r'(\d{1,2})/([\d]{1,2})', 'slash_format'),
            # [NEW] 일만 있는 경우: "14일", "25일에" (현재 월로 가정)
            (r'(\d{1,2})일(?:에|날)?', 'day_only'),
            
            # 상대 날짜
            (r'(오늘|내일|모레|글피)', 'relative_simple'),
            (r'(이번|다음|저번)\s*주?\s*(월|화|수|목|금|토|일)요일', 'relative_weekday'),
            # [NEW] "이번주 토요일" 패턴
            (r'이번\s*주\s*(토|일|월|화|수|목|금)', 'this_week_day'),
        ]
        
        # 영어 날짜 패턴
        self.english_date_patterns = [
            # "Dec 25", "December 25th, 2025"
            (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?', 'month_day'),
            # "12/25", "12/25/2025"
            (r'(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?', 'slash_format'),
            # "tomorrow", "today"
            (r'\b(today|tomorrow)\b', 'relative_simple'),
            # "next Friday", "this Monday"
            (r'\b(next|this)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'relative_weekday'),
        ]
        
        # 시간 패턴 (더 유연하게)
        self.time_patterns = [
            # 한국어 오전/오후: "오후 3시", "오전 10시 30분"
            (r'(오전|오후)\s*(\d{1,2})시(?:\s*(\d{1,2})분)?', 'korean_format'),
            # 12시간 형식: "3pm", "2:30 PM"
            (r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)\b', '12h_format'),
            # 24시간 형식: "14:30", "09:00"
            (r'\b([0-2]?\d):([0-5]\d)\b', '24h_format'),
            # [NEW] 한국어 유연 시간: "7시쯤", "3시에", "6시반"
            (r'(?<!오전|오후)(\d{1,2})시(?:쯤|경|에|반|부터|까지)?(?:\s*(\d{1,2})분)?', 'korean_flexible'),
            # [NEW] 저녁/아침/점심 시간대: "저녁 7시", "아침 8시"
            (r'(아침|점심|저녁|밤)\s*(\d{1,2})시', 'korean_period'),
        ]
        
        # 장소 패턴 (더 유연하게)
        self.location_patterns = [
            # [NEW] 역 + 출구 패턴: "홍대입구역 2번 출구", "강남역 3번출구"
            (r'([가-힣]+역)\s*(\d+번?\s*출구)?', 'korean_station'),
            # [NEW] "~에서 보자/만나자" 패턴
            (r'([가-힣]+(?:역|점|관|센터|카페|동|구|공원|광장|빌딩))\s*(?:\d+번?\s*출구)?(?:에서|에)', 'korean_location_at'),
            # 한국어: "강남역에서", "회의실에서"
            (r'([가-힣]{2,}(?:역|점|관|실|센터|빌딩|카페|식당))(?:에서|에)', 'korean_location'),
            # 영어: "at Seoul Station", "at home"
            (r'\bat\s+([A-Za-z][A-Za-z\s]+?)(?:\s+(?:at|on|in|with|for|and|,|\.)|$)', 'english_location'),
            # "@" 표기: "@강남역"
            (r'@\s*([가-힣A-Za-z\s]+)', 'at_symbol'),
            # [NEW] "홍대에서", "강남에서" (간단한 지역명)
            (r'([가-힣]{2,6})에서(?:\s|$)', 'korean_simple_location'),
        ]
        
        # 상대적 날짜 매핑
        self.relative_days = {
            '오늘': 0, 'today': 0,
            '내일': 1, 'tomorrow': 1,
            '모레': 2,
            '글피': 3,
        }
        
        # 요일 매핑
        self.weekday_map_kr = {
            '월': 0, '화': 1, '수': 2, '목': 3, 
            '금': 4, '토': 5, '일': 6
        }
        
        self.weekday_map_en = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        # 월 이름 매핑
        self.month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
    
    def extract_dates(self, text: str) -> List[str]:
        """날짜 추출 (YYYY-MM-DD 형식 또는 상대적 표현)"""
        results = []
        
        # 한국어 패턴 검사
        for pattern, pattern_type in self.korean_date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = self._parse_date_match(match, pattern_type, 'ko')
                if date_str:
                    results.append(date_str)
        
        # 영어 패턴 검사
        for pattern, pattern_type in self.english_date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = self._parse_date_match(match, pattern_type, 'en')
                if date_str:
                    results.append(date_str)
        
        return results
    
    def _parse_date_match(self, match: re.Match, pattern_type: str, lang: str) -> Optional[str]:
        """날짜 매칭 결과를 YYYY-MM-DD 형식으로 변환"""
        today = datetime.now()
        
        try:
            if pattern_type == 'absolute_full':
                # "2025년 1월 15일"
                year, month, day = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"
            
            elif pattern_type == 'absolute_short':
                # "12월 25일" (올해로 가정)
                month, day = match.groups()
                return f"{today.year}-{int(month):02d}-{int(day):02d}"
            
            elif pattern_type == 'day_only':
                # [NEW] "14일", "25일에" (현재 월로 가정)
                day = int(match.group(1))
                # 현재 날짜보다 이전이면 다음 달로
                if day < today.day:
                    next_month = today.month + 1
                    year = today.year
                    if next_month > 12:
                        next_month = 1
                        year += 1
                    return f"{year}-{next_month:02d}-{day:02d}"
                return f"{today.year}-{today.month:02d}-{day:02d}"
            
            elif pattern_type == 'this_week_day':
                # [NEW] "이번주 토요일" → 이번주 해당 요일
                weekday_char = match.group(1)
                target_weekday = self.weekday_map_kr[weekday_char]
                current_weekday = today.weekday()
                days_ahead = target_weekday - current_weekday
                if days_ahead < 0:
                    days_ahead += 7
                target_date = today + timedelta(days=days_ahead)
                return target_date.strftime("%Y-%m-%d")
            
            elif pattern_type == 'slash_format':
                # "12/25" 또는 "12/25/2025"
                if lang == 'ko':
                    month, day = match.groups()
                    return f"{today.year}-{int(month):02d}-{int(day):02d}"
                else:
                    groups = match.groups()
                    month, day = int(groups[0]), int(groups[1])
                    year = int(groups[2]) if groups[2] else today.year
                    if year < 100:  # 2자리 연도
                        year += 2000
                    return f"{year}-{month:02d}-{day:02d}"
            
            elif pattern_type == 'relative_simple':
                # "내일", "tomorrow"
                keyword = match.group(1).lower()
                days_offset = self.relative_days.get(keyword, 0)
                target_date = today + timedelta(days=days_offset)
                return target_date.strftime("%Y-%m-%d")
            
            elif pattern_type == 'relative_weekday':
                # "다음주 금요일", "next Friday"
                if lang == 'ko':
                    week_modifier, weekday = match.groups()
                    target_weekday = self.weekday_map_kr[weekday]
                else:
                    week_modifier, weekday = match.groups()
                    target_weekday = self.weekday_map_en[weekday.lower()]
                
                # 이번주/다음주 계산
                current_weekday = today.weekday()
                days_ahead = target_weekday - current_weekday
                
                if week_modifier in ['다음', 'next']:
                    if days_ahead <= 0:
                        days_ahead += 7
                    days_ahead += 7  # 다음주
                elif week_modifier in ['이번', 'this']:
                    if days_ahead < 0:
                        days_ahead += 7
                
                target_date = today + timedelta(days=days_ahead)
                return target_date.strftime("%Y-%m-%d")
            
            elif pattern_type == 'month_day':
                # "Dec 25" 또는 "December 25, 2025"
                month_str, day, year = match.groups()
                month = self.month_map[month_str.lower()[:3]]
                year = int(year) if year else today.year
                return f"{year}-{month:02d}-{int(day):02d}"
        
        except (ValueError, KeyError, AttributeError) as e:
            print(f"[Pattern Matcher] Date parse error: {e}")
            return None
        
        return None
    
    def extract_times(self, text: str) -> List[str]:
        """시간 추출 (HH:MM 형식) - 메시지 타임스탬프 필터링"""
        results = []
        seen = set()  # 중복 방지
        matched_positions = set()  # 매칭된 위치 추적
        
        # [NEW] 메시지 타임스탬프 위치 찾기 (오후 9:06 같은 패턴)
        # 이런 패턴은 줄 시작이나 끝에 있으면 메시지 타임스탬프임
        timestamp_positions = set()
        timestamp_pattern = r'(오전|오후)\s*\d{1,2}:\d{2}'
        for match in re.finditer(timestamp_pattern, text):
            # 줄 시작 또는 끝에 있으면 타임스탬프로 간주
            start = match.start()
            end = match.end()
            # 앞 문자가 줄바꿈이거나 시작이면 타임스탬프
            if start == 0 or text[start-1] in '\n\r':
                timestamp_positions.add((start, end))
            # 뒤 문자가 줄바꿈이거나 끝이면 타임스탬프
            elif end == len(text) or text[end:end+1] in '\n\r':
                timestamp_positions.add((start, end))
        
        # [NEW] 시간 패턴 우선순위 조정: 일정 시간 패턴 먼저 (7시쯤, 저녁 7시)
        priority_patterns = [
            # 한국어 유연 시간 (일정 관련): "7시쯤", "3시에", "6시반" - 최우선
            (r'(\d{1,2})시(?:쯤|경|에|반|부터|까지)', 'korean_flexible'),
            # 저녁/아침/점심 시간대: "저녁 7시", "아침 8시"
            (r'(아침|점심|저녁|밤)\s*(\d{1,2})시', 'korean_period'),
        ]
        
        # 먼저 우선순위 높은 패턴 처리
        for pattern, pattern_type in priority_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()
                # 타임스탬프 위치와 겹치면 스킵
                if any(ts <= start < te or ts < end <= te for ts, te in timestamp_positions):
                    continue
                if any(start <= pos < end or pos <= start < pos_end for pos, pos_end in matched_positions):
                    continue
                
                time_str = self._parse_time_match(match, pattern_type)
                if time_str and time_str not in seen:
                    results.append(time_str)
                    seen.add(time_str)
                    matched_positions.add((start, end))
        
        # 기존 패턴도 처리 (우선순위 패턴으로 이미 찾은 것 제외)
        for pattern, pattern_type in self.time_patterns:
            # 이미 우선순위 패턴에서 처리한 타입은 스킵
            if pattern_type in ['korean_flexible', 'korean_period']:
                continue
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()
                
                # 타임스탬프 위치와 겹치면 스킵
                if any(ts <= start < te or ts < end <= te for ts, te in timestamp_positions):
                    continue
                
                # 이미 매칭된 위치와 겹치는지 확인
                if any(start <= pos < end or pos <= start < pos_end for pos, pos_end in matched_positions):
                    continue
                
                time_str = self._parse_time_match(match, pattern_type)
                if time_str and time_str not in seen:
                    results.append(time_str)
                    seen.add(time_str)
                    matched_positions.add((start, end))
        
        return results
    
    def _parse_time_match(self, match: re.Match, pattern_type: str) -> Optional[str]:
        """시간 매칭 결과를 HH:MM 형식으로 변환"""
        try:
            if pattern_type == '24h_format':
                # "14:30"
                hour, minute = match.groups()
                hour, minute = int(hour), int(minute)
                # 시간 범위 검증 (0-23시, 0-59분)
                if hour > 23 or minute > 59:
                    return None
                return f"{hour:02d}:{minute:02d}"
            
            elif pattern_type == '12h_format':
                # "3pm", "2:30 PM"
                hour, minute, meridiem = match.groups()
                hour = int(hour)
                minute = int(minute) if minute else 0
                
                if meridiem.lower() == 'pm' and hour != 12:
                    hour += 12
                elif meridiem.lower() == 'am' and hour == 12:
                    hour = 0
                
                return f"{hour:02d}:{minute:02d}"
            
            elif pattern_type == 'korean_format':
                # "오후 3시", "오전 10시 30분"
                meridiem, hour, minute = match.groups()
                hour = int(hour)
                minute = int(minute) if minute else 0
                
                if meridiem == '오후' and hour != 12:
                    hour += 12
                elif meridiem == '오전' and hour == 12:
                    hour = 0
                
                return f"{hour:02d}:{minute:02d}"
            
            elif pattern_type == 'korean_simple':
                # "14시", "3시 30분"
                hour, minute = match.groups()
                hour = int(hour)
                minute = int(minute) if minute else 0
                return f"{hour:02d}:{minute:02d}"
            
            elif pattern_type == 'korean_flexible':
                # [NEW] "7시쯤", "3시에", "6시반"
                hour, minute = match.groups()
                hour = int(hour)
                minute = int(minute) if minute else 0
                # "반" 처리는 정규식에서 이미 제외됨
                return f"{hour:02d}:{minute:02d}"
            
            elif pattern_type == 'korean_period':
                # [NEW] "저녁 7시" → 저녁이면 12시간 더함
                period, hour = match.groups()
                hour = int(hour)
                # 저녁/밤이면 오후로 처리 (12시간제 → 24시간제)
                if period in ['저녁', '밤'] and hour < 12:
                    hour += 12
                return f"{hour:02d}:00"
        
        except (ValueError, AttributeError) as e:
            print(f"[Pattern Matcher] Time parse error: {e}")
            return None
        
        return None
    
    def extract_locations(self, text: str) -> Optional[str]:
        """장소 추출 - 더 유연하게"""
        for pattern, pattern_type in self.location_patterns:
            match = re.search(pattern, text)
            if match:
                if pattern_type == 'korean_station':
                    # "홍대입구역 2번 출구" → 전체 반환
                    station = match.group(1).strip()
                    exit_info = match.group(2).strip() if match.group(2) else ""
                    if exit_info:
                        return f"{station} {exit_info}"
                    return station
                elif pattern_type == 'korean_location_at':
                    # "홍대입구역에서" → 역 + 출구 정보
                    location = match.group(0)
                    # 조사 제거
                    location = re.sub(r'(에서|에)$', '', location).strip()
                    return location
                else:
                    location = match.group(1).strip()
                    return location
        
        return None


# 테스트 코드
if __name__ == "__main__":
    matcher = SchedulePatternMatcher()
    
    test_cases = [
        "내일 오후 3시에 강남역에서 미팅",
        "12월 25일 크리스마스 파티",
        "다음주 금요일 오전 10시 회의",
        "Meeting at 3pm tomorrow",
        "Dec 25 Christmas dinner at home",
    ]
    
    for text in test_cases:
        print(f"\n입력: {text}")
        print(f"  날짜: {matcher.extract_dates(text)}")
        print(f"  시간: {matcher.extract_times(text)}")
        print(f"  장소: {matcher.extract_locations(text)}")
