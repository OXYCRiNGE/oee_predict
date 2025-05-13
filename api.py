"""
API для прогнозирования OEE (Общая эффективность оборудования).

Назначение:
Данный API предоставляет возможность генерации прогнозов OEE на основе исторических данных, 
хранящихся в Excel-файле (по умолчанию 'dataset/signal_data_full.xlsx'). API использует 
предобученную модель машинного обучения (CatBoost) или обучает новую, если модель отсутствует. 
Основной функционал доступен через GET-запрос к эндпоинту /api/get/forecast, который возвращает 
прогноз OEE на указанное количество дней (от 1 до 60) в формате JSON. 

Основные задачи API:
- Загрузка и предобработка данных.
- Загрузка или обучение модели прогнозирования.
- Генерация прогнозов OEE на заданный горизонт.
- Обработка ошибок и логирование всех операций.

Логирование осуществляется в единый файл logs/app.log, что обеспечивает централизованный сбор 
информации о работе API.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from logging_config import setup_logging
from database import init_db_engine, get_forecast_from_db
import logging
from pathlib import Path

# Настройка логирования
setup_logging()

# Конфигурация
load_dotenv()
DATA_FILE = Path(os.getenv('DATA_FILE', 'dataset/signal_data_full.xlsx'))
MODEL_FILE = Path(os.getenv('MODEL_FILE', 'model/best_pipeline_model.pkl'))

# Lifespan-обработчик
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Запуск приложения")
    init_db_engine()  # Инициализация движка SQLAlchemy
    yield
    logging.info("Остановка приложения")

# Инициализация FastAPI
app = FastAPI(
    title="API прогноза OEE",
    description="API для генерации прогнозов OEE",
    lifespan=lifespan
)

# Обработчик ошибок валидации
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logging.error(f"Ошибка валидации: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Параметр days должен быть целым числом от 1 до 60"}
    )

@app.get("/forecast", summary="Генерация прогноза OEE", response_description="Данные прогноза в формате JSON")
async def get_forecast(days: int = Query(60, ge=1, le=60)):
    """
    Возвращает прогноз OEE на указанное количество дней из базы данных или генерирует новый.
    - **days**: Количество дней для прогноза (1-60, по умолчанию=60)
    """
    logging.info(f"Получен запрос: days={days}")
    try:
        forecast_df = get_forecast_from_db(days)
        # logging.info("Прогноз взят из таблицы current_forecast")
        forecast_json = forecast_df.to_dict(orient='records')
        logging.info("Запрос успешно обработан")
        return {"forecast": forecast_json}
    except FileNotFoundError as e:
        logging.error(f"Файл не найден: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Файл не найден: {str(e)}")
    except Exception as e:
        logging.error(f"Ошибка обработки запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")