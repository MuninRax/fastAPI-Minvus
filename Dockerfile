# ใช้ official Python base image ที่รองรับ FastAPI และ TensorFlow
FROM python:3.10

# ตั้งค่า working directory ภายใน container
WORKDIR /app

# คัดลอกไฟล์ requirements.txt ไปที่ container
COPY requirements-for-container.txt .

# ติดตั้ง dependencies
RUN pip install -r requirements-for-container.txt

# คัดลอกโค้ดทั้งหมดไปยัง container
COPY . .

# # ระบุคำสั่งรัน FastAPI ด้วย Uvicorn
# CMD ["uvicorn", "main:app", "--reload"]
