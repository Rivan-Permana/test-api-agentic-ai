# --- Dockerfile ---
# Perintah untuk membangun image:
# > docker build -t ml-bi-pipeline .
#
# Perintah untuk menjalankan container secara lokal:
# 1. Buat file .env dengan rahasia Anda (misal: OPENAI_API_KEY=sk-...)
# 2. > docker run --env-file .env -p 8080:8080 ml-bi-pipeline
#
# Contoh pengujian dengan cURL:
# > curl -X POST "http://localhost:8080/analyze" \
#   -H "accept: application/json" \
#   -H "Content-Type: multipart/form-data" \
#   -F "prompt=Berapa pendapatan rata-rata per bulan?" \
#   -F "file=@/path/to/your/data.csv"

# Gunakan runtime Python 3.10-slim sebagai base image
FROM python:3.10-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk caching layer
COPY requirements.txt .

# --- PERUBAHAN DI SINI ---
# 1. Instal numpy terlebih dahulu karena beberapa paket (seperti duckdb) membutuhkannya saat instalasi.
RUN pip install --no-cache-dir numpy

# 2. Instal sisa dependensi Python dari requirements.txt
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt
# --- AKHIR PERUBAHAN ---

# Salin sisa kode aplikasi
COPY main.py .

# Tetapkan PORT yang akan diekspos oleh container
# Cloud Run akan menggunakan variabel ini. Default ke 8080.
ENV PORT 8080

# Jalankan aplikasi menggunakan Gunicorn saat container dimulai
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
