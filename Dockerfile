FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgl1 libglib2.0-0 tesseract-ocr && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "ai_candlestick_trader.api:app", "--host", "0.0.0.0", "--port", "8000"]
