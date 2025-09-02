# --- Dockerfile ---
# Gunakan runtime Python 3.10-slim sebagai base image
FROM python:3.10-slim

# --- PERUBAHAN DI SINI ---
# Instal system dependencies yang dibutuhkan untuk build paket dari source
# seperti cmake untuk duckdb
RUN apt-get update && apt-get install -y cmake build-essential
# --- AKHIR PERUBAHAN ---

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk caching layer
COPY requirements.txt .

# 1. Instal numpy terlebih dahulu
RUN pip install --no-cache-dir numpy

# 2. Instal sisa dependensi Python dari requirements.txt
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Salin sisa kode aplikasi
COPY main.py .

# Tetapkan PORT yang akan diekspos oleh container
ENV PORT 8081

# Jalankan aplikasi menggunakan Gunicorn saat container dimulai
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "127.0.0.1:8081", "--timeout", "120"]
