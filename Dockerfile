FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY . .

# Создаём директории
RUN mkdir -p /app/dataset /app/model /app/logs /app/data /app/backup

# Открываем порт для FastAPI
EXPOSE 8000

# Команда по умолчанию для API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]