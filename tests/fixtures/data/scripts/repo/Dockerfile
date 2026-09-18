FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m compileall /app
CMD ["python", "app.py"]
