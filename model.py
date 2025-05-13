import warnings
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
import holidays
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from scipy.interpolate import interp1d
from catboost import CatBoostRegressor
import logging
from dotenv import load_dotenv
from logging_config import setup_logging
from pathlib import Path

# Настройка логирования
setup_logging()

# Конфигурация
load_dotenv()
DATA_FILE = Path(os.getenv('DATA_FILE', 'dataset/signal_data_full.xlsx'))

# Игнорирование предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def preprocess_data(data, target='OEE', exog_columns=None, test_size=0.2):
    data = pd.DataFrame(data.copy())
    data.fillna(0, inplace=True)
    data.drop(['Объект'], axis=1, inplace=True, errors='ignore')
    data['Дата'] = pd.to_datetime(data['Дата']).dt.normalize()
    data.set_index('Дата', inplace=True)
    data = data.groupby('Дата').sum()

    # Вычисление OEE
    data["ВП"] = data["Серийное производство"] + data["Программа выполняется"]
    data["ВРО"] = data["ВП"] + data["Прогрев станка"] + data["Отработка программы"] + \
                  data["Ручной режим"] + data["Станок включен"]
    data["ВРП"] = data["ВРО"] + data["Наладка"] + data["Контроль ОТК"] + \
                  data["Регламентированный перерыв"] + data["Уборка оборудования"] + \
                  data["Сервисное обслуживание"] + data["Отсутствие заготовки"] + \
                  data["Отсутствие программы"] + data["Отсутствие инструмента"] + \
                  data["Отсутствие КД/модели"] + data["Ремонтные работы"] + \
                  data["Авария"] + data["Аварийная остановка"]
    data["OEE"] = data["ВРО"] / data["ВРП"] * data["ВП"] / data["ВРО"] * 0.95

    # Удаление ненужных столбцов
    exclude_columns = ["Производительность", "Доступность", "ВРО", "ВРП", "ВП"]
    data.drop(exclude_columns, axis=1, inplace=True, errors="ignore")

    # Удаление столбцов с большим количеством NaN или нулей
    threshold = len(data) * 0.7
    threshold_columns = [
        col for col in data.columns
        if data[col].isna().sum() > threshold or (data[col] == 0).sum() > threshold
    ]
    data = data.drop(columns=threshold_columns)

    # Применение PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    data[data.drop(["OEE"], axis=1).columns] = pt.fit_transform(data.drop(["OEE"], axis=1))

# Лаговые признаки
    lag_steps = [1, 7, 14, 30, 60]
    for lag in lag_steps:
        data[f"{target}_lag_{lag}"] = data[target].shift(lag)
        if exog_columns is not None:
            for col in exog_columns:
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        else:
            for col in data.drop(target, axis=1).columns.to_list():
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)

    # Скользящие средние и стандартное отклонение
    window_sizes = [7, 14, 30, 60]
    for window in window_sizes:
        data[f"{target}_mean_{window}"] = data[target].rolling(window=window).mean()
        data[f"{target}_std_{window}"] = data[target].rolling(window=window).std()
        if exog_columns is not None:
            for col in exog_columns:
                data[f"{col}_mean_{window}"] = data[col].rolling(window=window).mean()
                data[f"{col}_std_{window}"] = data[col].rolling(window=window).std()

    # Временные признаки
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["day_of_year"] = data.index.dayofyear
    data["week_of_year"] = data.index.isocalendar().week
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # Праздники
    ru_holidays = holidays.RU(years=data.index.year.unique())
    data["holiday"] = data.index.map(lambda x: 1 if x in ru_holidays else 0)

    # Удаляем NaN
    data.dropna(inplace=True)

    # Разделение на train/test
    size = int(len(data) * (1 - test_size))
    X_train = data.iloc[:size].drop(target, axis=1)
    X_test = data.iloc[size:].drop(target, axis=1)
    y_train = data[target].iloc[:size]
    y_test = data[target].iloc[size:]

    return X_train, X_test, y_train, y_test,

def save_model(model, model_file):
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    with open(model_file, 'wb') as f:
        joblib.dump(model, f)
    logging.info(f"Модель сохранена в {model_file}")

def train_model(X_train, X_test, y_train, y_test, model_file=Path('model/best_pipeline_model.pkl')):
    bp_model = {
        'learning_rate': 0.05,
        'l2_leaf_reg': 5,
        'iterations': 300,
        'depth': 4
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", CatBoostRegressor(
            **bp_model, 
            verbose=0, 
            random_seed=42, 
            has_time=True,
            thread_count=-1
            )
        )
    ])
    pipeline.fit(X_train, y_train)
            
    # Вычисление метрик на тестовых данных

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Логирование метрик
    logging.info(f"Метрики модели на тестовых данных: MAE={mae:.2f} MSE={mse:.2f} RMSE={rmse:.2f} MAPE={mape:.2%} R2={r2:.2%}")
    
    save_model(pipeline, model_file)
    return pipeline

def forecast_future(model, X_test, y_test, forecast_horizon=60):
    if not hasattr(model, 'predict'):
        raise ValueError(f"Объект не является моделью: {type(model)}")
    future_dates = pd.date_range(start=y_test.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    last_known_data = X_test.iloc[-1:].copy()
    future_preds = []

    for _ in range(forecast_horizon):
        next_pred = model.predict(last_known_data)[0]
        future_preds.append(next_pred)
        last_known_data = last_known_data.shift(-1, axis=1)
        last_known_data.iloc[:, -1] = next_pred

    forecast_df = pd.DataFrame({"date": future_dates, "oee": future_preds})
    return forecast_df

def save_forecast(forecast_df, output_file='forecast_60_days.xlsx'):
    forecast_df.to_excel(output_file, index=False)
    return output_file