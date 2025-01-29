# fastAPI-Milvus

- เอาไฟล์ flattened_price_list.json มาใส่
- pip install -r requirement.txt
- run uvicorn main:app --reload

# Docker Container
- run `docker-compose up --build`
- ช่วงนี้จะนานพอสมควร เนื่องจาก install requirements
- เมื่อเสร็จแล้ว Log จะแจ้งว่า `INFO:     Application startup complete.`

### !!!
- ถ้ามี collection ชื่อเดียวกันบน Milvus จะ error ต้องลบหรือแก้ชื่อในไฟล์ config.py ก่อน หรือไม่ก็แก้โค้ดให้ check


