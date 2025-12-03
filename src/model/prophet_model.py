# src/model/prophet_model.py

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

# Абсолютный путь к корневой директории проекта
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def train_prophet(days=30):
    """
    Обучает Prophet и создаёт:
    - прогноз
    - метрики (MAE, RMSE) по последним реальным данным
    - файл прогноза
    """

    # ====== 1. Загружаем данные ======
    data_path = os.path.join(BASE, "data/processed/usd_preprocessed.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.rename(columns={"date": "ds", "rate": "y"})

    # Последняя дата данных (у тебя это 2025-12-03)
    last_date = df["ds"].max()

    print("Последняя дата данных:", last_date)

    # ====== 2. Обучаем Prophet на ВСЕХ данных ======
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df)

    # ====== 3. METRICS: считаем MAE/RMSE только по последним 30 дням ======
    df_test = df.tail(30)
    future_test = df_test[["ds"]]
    fc_test = model.predict(future_test)

    y_true = df_test["y"].values
    y_pred = fc_test["yhat"].values

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {"mae": mae, "rmse": rmse}

    # ====== 4. Формируем будущий прогноз ======
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Оставляем только будущее (начиная со следующего дня)
    forecast = forecast[forecast["ds"] > last_date]

    # Переводим в итоговый DataFrame
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    out = out.rename(columns={
        "ds": "date",
        "yhat": "forecast",
        "yhat_lower": "lower",
        "yhat_upper": "upper"
    })

    # ====== 5. Сохраняем прогноз ======
    out_path = os.path.join(BASE, "data/processed/usd_prophet_forecast.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    # ====== 6. Сохраняем метрики ======
    metrics_path = os.path.join(BASE, "models/prophet_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    pd.DataFrame([metrics]).to_json(metrics_path, orient="records")

    print("Prophet готов:")
    print("Прогноз сохранён в:", out_path)
    print("Метрики сохранены:", metrics_path)

    return out, metrics
