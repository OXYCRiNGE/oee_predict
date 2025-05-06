import schedule
import time
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from winnum import process_signal_data
from model import preprocess_data, train_model
from logging_config import setup_logging
from pathlib import Path

# Настройка логирования
setup_logging()

# Конфигурация
load_dotenv()
DATA_FILE = Path(os.getenv('DATA_FILE', 'dataset/signal_data_full.xlsx'))
MODEL_FILE = Path(os.getenv('MODEL_FILE', 'model/main_pipeline_model.pkl'))

def job_extract_data():
    try:
        logging.info("Начало извлечения данных")
        df_tags = pd.read_excel(Path('data/tags.xlsx'))
        df_uuid = pd.read_excel(Path('data/dmg_uuid_full.xlsx'))
        process_signal_data(df_uuid, df_tags)
        logging.info("Извлечение данных завершено")
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных: {str(e)}")

def job_train():
    try:
        logging.info("Начало обучения модели")
        df = pd.read_excel(DATA_FILE)
        X_train, _, y_train, _ = preprocess_data(df)
        model = train_model(X_train, y_train, model_file=MODEL_FILE)
        if not hasattr(model, 'predict'):
            raise ValueError(f"Обученный объект не является моделью: {type(model)}")
        logging.info("Обучение модели завершено")
    except Exception as e:
        logging.error(f"Ошибка при обучении: {str(e)}")

# Планирование задач
schedule.every().day.at("00:30").do(job_extract_data)
schedule.every().day.at("01:00").do(job_train)

if __name__ == "__main__":
    logging.info("Планировщик запущен")
    job_extract_data()
    job_train()
    while True:
        schedule.run_pending()
        time.sleep(60)