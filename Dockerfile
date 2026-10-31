FROM python:3.10-slim@sha256:a45c323edaa44976ef63b9a85e0d3bd7bbf31676029dccfbc119f88a65311852

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir "typing_extensions>=4.10.0" \
    && pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
