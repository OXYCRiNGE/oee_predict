import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, inspect, text, MetaData, Table, Column, Integer, Date, Float, DateTime, ForeignKey
from sqlalchemy.exc import OperationalError
from psycopg import Connection
from logging_config import setup_logging
import env
import time

# Настройка логирования
setup_logging()

DB_HOST = env.DB_HOST
DB_PORT = env.DB_PORT
DB_USER = env.DB_USER
DB_PASSWORD = env.DB_PASSWORD
DB_NAME = env.DB_NAME

# Создание строки подключения
CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
CONNECTION_STRING_DEFAULT = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"

# Инициализация движка
engine = None

def clean_string(s):
    """Преобразование строки в UTF-8 с заменой некорректных символов."""
    if not isinstance(s, str):
        s = str(s)
    return s.encode('utf-8', errors='replace').decode('utf-8')

def wait_for_postgres(max_attempts=20, delay=5):
    """Ожидание готовности PostgreSQL."""
    default_engine = create_engine(CONNECTION_STRING_DEFAULT, echo=False)
    attempts = 0
    while attempts < max_attempts:
        try:
            with default_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logging.info("PostgreSQL готов к подключению")
            default_engine.dispose()
            return True
        except OperationalError as e:
            logging.warning(f"PostgreSQL не готов, попытка {attempts + 1}/{max_attempts}: {str(e)}")
            attempts += 1
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Неожиданная ошибка при проверке PostgreSQL: {str(e)}")
            attempts += 1
            time.sleep(delay)
    logging.error("Не удалось подключиться к PostgreSQL после максимального количества попыток")
    default_engine.dispose()
    return False

def create_database():
    """Создание базы данных oee_predict, если она не существует."""
    if not wait_for_postgres():
        raise Exception("PostgreSQL недоступен, не удалось создать базу данных")
    try:
        # Подключение к серверу PostgreSQL (база по умолчанию 'postgres') через psycopg напрямую
        conn = Connection.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname='postgres',
            autocommit=True
        )
        cursor = conn.cursor()

        # Проверка существования базы данных
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logging.info(f"База данных {DB_NAME} создана")
        else:
            logging.info(f"База данных {DB_NAME} уже существует")

        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Ошибка при создании базы данных: {str(e)}")
        raise

def init_db_engine():
    """Инициализация движка SQLAlchemy для базы данных oee_predict."""
    global engine
    if not wait_for_postgres():
        raise Exception("PostgreSQL недоступен, не удалось инициализировать движок")
    try:
        engine = create_engine(CONNECTION_STRING, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Движок SQLAlchemy инициализирован")
    except Exception as e:
        logging.error(f"Ошибка инициализации движка SQLAlchemy: {str(e)}")
        raise

def create_tables():
    """Создание таблиц call, forecast и current_forecast, если они не существуют."""
    try:
        inspector = inspect(engine)
        call_exists = 'call' in inspector.get_table_names()
        forecast_exists = 'forecast' in inspector.get_table_names()
        current_forecast_exists = 'current_forecast' in inspector.get_table_names()

        metadata = MetaData()

        if not call_exists:
            Table(
                'call', metadata,
                Column('call_id', Integer, primary_key=True, autoincrement=True),
                Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP'))
            )
            logging.info("Таблица call создана")
        else:
            logging.info("Таблица call уже существует")

        if not forecast_exists:
            Table(
                'forecast', metadata,
                Column('id', Integer, primary_key=True),
                Column('call_id', Integer, ForeignKey('call.call_id', ondelete='CASCADE'), nullable=False),
                Column('forecast_date', Date, nullable=False),
                Column('oee', Float, nullable=False)
            )
            logging.info("Таблица forecast создана")
        else:
            logging.info("Таблица forecast уже существует")

        if not current_forecast_exists:
            Table(
                'current_forecast', metadata,
                Column('id', Integer, primary_key=True),
                Column('forecast_date', Date, nullable=False),
                Column('oee', Float, nullable=False)
            )
            logging.info("Таблица current_forecast создана")
        else:
            logging.info("Таблица current_forecast уже существует")

        # Создание всех таблиц
        metadata.create_all(engine)

        # Создание индексов после создания таблиц
        with engine.connect() as conn:
            if not forecast_exists:
                conn.execute(text("CREATE INDEX idx_forecast_call_id ON forecast (call_id)"))
                conn.execute(text("CREATE INDEX idx_forecast_date ON forecast (forecast_date)"))
                logging.info("Индексы для таблицы forecast созданы")
            if not current_forecast_exists:
                conn.execute(text("CREATE INDEX idx_current_forecast_date ON current_forecast (forecast_date)"))
                logging.info("Индекс для таблицы current_forecast создан")
            conn.commit()

    except Exception as e:
        logging.error(f"Ошибка при создании таблиц: {str(e)}")
        raise

def save_forecast_to_db(forecast_df):
    """Сохранение прогнозов в таблицы forecast и current_forecast."""
    try:
        # Создание записи в таблице call и получение call_id
        with engine.connect() as conn:
            result = conn.execute(
                text("INSERT INTO call (created_at) VALUES (:created_at) RETURNING call_id"),
                {"created_at": datetime.now()}
            )
            call_id = result.fetchone()[0]
            conn.commit()
        logging.info(f"Запись в таблице call создана с call_id={call_id}")

        # Подготовка DataFrame для forecast
        df_forecast = forecast_df[['date', 'oee']].copy()
        df_forecast['call_id'] = call_id
        df_forecast = df_forecast.rename(columns={'date': 'forecast_date'})
        # logging.info(f"Данные для записи в forecast: {df_forecast.head().to_dict()}")

        # Сохранение в таблицу forecast
        start_time = time.time()
        df_forecast.to_sql('forecast', con=engine, index=False, if_exists='append')
        logging.info(f"Сохранено {len(df_forecast)} записей в forecast за {time.time() - start_time:.2f} секунд с call_id={call_id}")

        # Очистка и сохранение в таблицу current_forecast
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM current_forecast"))
            conn.commit()
        logging.info("Таблица current_forecast очищена")

        df_current = forecast_df[['date', 'oee']].copy()
        df_current = df_current.rename(columns={'date': 'forecast_date'})
        # logging.info(f"Данные для записи в current_forecast: {df_current.head().to_dict()}")

        start_time = time.time()
        df_current.to_sql('current_forecast', con=engine, index=False, if_exists='append')
        logging.info(f"Сохранено {len(df_current)} записей в current_forecast за {time.time() - start_time:.2f} секунд")

        return call_id
    except Exception as e:
        logging.error(f"Ошибка при сохранении прогнозов: {str(e)}")
        raise

def get_forecast_from_db(days: int):
    """Получение прогнозов из таблицы current_forecast за указанное количество дней."""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT forecast_date, oee 
                FROM current_forecast
                ORDER BY forecast_date 
                LIMIT :days
            """)
            result = conn.execute(query, {"days": days})
            rows = result.fetchall()
            if not rows:
                return None
            forecast_df = pd.DataFrame(rows, columns=['date', 'oee'])
            return forecast_df
    except Exception as e:
        logging.error(f"Ошибка при получении прогнозов: {str(e)}")
        return None