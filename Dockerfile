# Use a base image compatible with your Pi (ARM64) and local dev (x86)
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /solar-forecasting

# deps first for layer caching
COPY requirements.txt . 
RUN apt-get update && apt-get install -y sqlite3 \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# app code
COPY . .

EXPOSE 5000

# run the Flask app (make sure app.app calls app.run(host="0.0.0.0", port=5000))
CMD ["python", "-m", "app.app"]
