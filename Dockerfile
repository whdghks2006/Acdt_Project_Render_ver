# ============================================
# Stage 1: Builder - Dependencies 설치
# ============================================
FROM python:3.10-slim AS builder

WORKDIR /code

# 시스템 의존성 설치 (빌드용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사 (레이어 캐싱 최적화)
COPY requirements.txt .

# Python 패키지 설치 (가상환경에 설치하여 복사 용이하게)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime - 최종 이미지
# ============================================
FROM python:3.10-slim

WORKDIR /code

# 런타임 시스템 의존성 (최소한만 설치)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Builder에서 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 코드 복사
COPY . .

# 캐시 디렉토리 설정
RUN mkdir -p /code/cache && \
    chmod -R 777 /code/cache

ENV HF_HOME=/code/cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 포트 노출
EXPOSE 7860

# 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]