# Python 3.9 기반 이미지 사용
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libgl1-mesa-glx \    
    libglib2.0-0 \       
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart
# 애플리케이션 코드 복사
COPY . .

EXPOSE 8000
# Uvicorn을 사용하여 FastAPI 애플리케이션 실행
CMD ["uvicorn", "TossBenefaceFastApi:app", "--host", "0.0.0.0", "--port", "8000"]