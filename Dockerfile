FROM python:3.11-slim

WORKDIR /Classification_Deep_Learning_project

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3","app.py"]