FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN mkdir -p data artifacts reports
EXPOSE 8501
CMD ["sh", "-c", "python generate_data.py && python -m src.train_model && streamlit run dashboard.py --server.address=0.0.0.0 --server.port=8501"]
