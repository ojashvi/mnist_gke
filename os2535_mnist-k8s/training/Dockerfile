
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY model.py .
COPY train.py .

ENV MODEL_DIR=/models
ENV DATA_DIR=/tmp/mnist

CMD ["python", "train.py"]
