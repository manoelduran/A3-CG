
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
  apt-get install -y \
  libgl1 \
  libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "worker.py"]



