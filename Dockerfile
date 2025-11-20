# 1. Python 3.10 버전을 기반으로 시작
FROM python:3.10

# 2. 작업 디렉토리 설정
WORKDIR /code

# 3. requirements.txt 복사 및 설치
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. 나머지 모든 파일 복사
COPY . .

# 5. 권한 설정 (Hugging Face 권장사항: 캐시 폴더 쓰기 권한)
RUN mkdir -p /code/cache
ENV TRANSFORMERS_CACHE=/code/cache
ENV HF_HOME=/code/cache
RUN chmod -R 777 /code

# 6. 서버 실행 (Hugging Face는 무조건 7860 포트를 사용해야 함!)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]