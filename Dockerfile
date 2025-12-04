FROM python:3.11

# Attached information
LABEL author.name="HOAINAM" \
    author.email="namlh@dgk.vn"

ENV TZ="Asia/Bangkok"

# --- ĐÃ SỬA: Thay libgl1-mesa-glx bằng libgl1 ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# ------------------------------------------------

RUN mkdir /www
WORKDIR /www
COPY requirements.txt /www/

# Cài đặt python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . /www