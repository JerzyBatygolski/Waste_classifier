FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV API_URL="https://waste-classifier-api-spsy2vhjeq-lm.a.run.app"

EXPOSE 8080

CMD ["streamlit", "run", "app.py", \
     "--server.port", "8080", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]