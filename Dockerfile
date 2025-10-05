FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "120"]
