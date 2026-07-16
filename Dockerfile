# ─── GARCH-Guided LSTM — Docker image for Render / any Docker host ───
FROM python:3.10-slim

# arch, and other scientific libs need a C compiler to build from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Render (and most Docker hosts) inject the port to bind to via $PORT
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]