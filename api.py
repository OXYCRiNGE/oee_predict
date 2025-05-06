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
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from model import preprocess_data, forecast_future, train_model
from logging_config import setup_logging
import logging
from pathlib import Path

# Настройка логирования
setup_logging()

# Конфигурация
load_dotenv()
DATA_FILE = Path(os.getenv('DATA_FILE', 'dataset/signal_data_full.xlsx'))
MODEL_FILE = Path(os.getenv('MODEL_FILE', 'model/best_pipeline_model.pkl'))

# Lifespan-обработчик (без инициализации)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Запуск приложения")
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
    Генерирует прогноз OEE на указанное количество дней.
    - **days**: Количество дней для прогноза (1-60, по умолчанию=60)
    """
    logging.info(f"Получен запрос: days={days}")
    try:
        # Загрузка данных
        # logging.info(f"Загрузка данных из {DATA_FILE}")
        df = pd.read_excel(DATA_FILE)
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # Загрузка или обучение модели
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        if not os.path.exists(MODEL_FILE):
            logging.info(f"Модель {MODEL_FILE} не найдена. Обучение новой модели.")
            model = train_model(X_train, y_train, MODEL_FILE)
        else:
            # logging.info(f"Загрузка модели из {MODEL_FILE}")
            model = joblib.load(MODEL_FILE)
            if not hasattr(model, 'predict'):
                logging.error(f"Загруженный объект не является моделью: {type(model)}")
                raise HTTPException(status_code=500, detail="Загруженный объект не является моделью")

        # Генерация прогноза
        logging.info(f"Генерация прогноза на {days} дней")
        forecast_df = forecast_future(model, X_test, y_test, forecast_horizon=days)
        forecast_json = forecast_df.to_dict(orient='records')

        logging.info("Запрос успешно обработан")
        return {"forecast": forecast_json}
    except FileNotFoundError as e:
        logging.error(f"Файл не найден: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Файл не найден: {str(e)}")
    except Exception as e:
        logging.error(f"Ошибка обработки запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")