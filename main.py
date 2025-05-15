import schedule
import time
import pandas as pd
import logging
import env
from winnum import process_signal_data
from model import preprocess_data, train_model, forecast_future
from logging_config import setup_logging
from database import create_database, init_db_engine, create_tables, save_forecast_to_db
from pathlib import Path
import pytz

# Настройка логирования
setup_logging()


DATA_FILE = env.DATA_FILE
MODEL_FILE = env.MODEL_FILE
START_WITH_DOWNLOAD = env.START_WITH_DOWNLOAD

TZ = pytz.timezone('Europe/Moscow')


def job_extract_data():
    try:
        logging.info("Начало извлечения данных")
        df_tags = pd.read_excel(Path('data/tags.xlsx'), engine='openpyxl')
        df_uuid = pd.read_excel(Path('data/dmg_uuid_full.xlsx'), engine='openpyxl')
        process_signal_data(df_uuid, df_tags)
        logging.info("Извлечение данных завершено")
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных: {str(e)}")

def job_train_and_forecast():
    try:
        logging.info("Начало обучения модели и генерации прогноза")
        df = pd.read_excel(DATA_FILE, engine='openpyxl')
        X_train, X_test, y_train, y_test = preprocess_data(df, exog_columns=['OEE'])
        
        # Обучение модели
        model = train_model(X_train, X_test, y_train, y_test, model_file=MODEL_FILE)
        if not hasattr(model, 'predict'):
            raise ValueError(f"Обученный объект не является моделью: {type(model)}")
        
        # Генерация прогноза
        forecast_df = forecast_future(model, X_test, y_test, forecast_horizon=60)
        
        # Сохранение прогноза в базу данных
        call_id = save_forecast_to_db(forecast_df)
        
        logging.info(f"Обучение модели и генерация прогноза завершены, call_id={call_id}")
    except Exception as e:
        logging.error(f"Ошибка при обучении или прогнозировании: {str(e)}")

# Проверка и создание базы данных и таблиц при запуске
try:
    create_database()
    init_db_engine()
    create_tables()
except Exception as e:
    logging.error(f"Ошибка при инициализации базы данных: {str(e)}")
    exit(1)

# Планирование задач
schedule.every().day.at("03:30", "Europe/Moscow").do(job_extract_data)
schedule.every().day.at("04:00", "Europe/Moscow").do(job_train_and_forecast)

if __name__ == "__main__":
    logging.info("Планировщик запущен")
    if START_WITH_DOWNLOAD:
        job_extract_data()
        job_train_and_forecast()
    while True:
        schedule.run_pending()
        time.sleep(60)