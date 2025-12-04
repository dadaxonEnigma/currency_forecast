# src/model/prophet_model.py
"""
Модуль обучения и прогнозирования модели Prophet для курса USD → UZS.

Функции:
    • обучение модели Prophet на всей истории
    • backtesting — оценка на последних 30 днях
    • сохранение backtest предсказаний
    • генерация прогноза на N будущих дней
    • сохранение прогноза и метрик

Файлы:
    data/processed/prophet_test_predictions.csv      ← прогноз vs реальность (30 дней)
    data/processed/usd_prophet_forecast.csv          ← будущий прогноз
    models/prophet_metrics.json                      ← метрики MAE, RMSE
"""

import os
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Абсолютный путь к проекту
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def train_prophet(days: int = 90):
    """
    Обучает Prophet, формирует:
        • backtest на последних 30 днях
        • прогноз на будущие дни
        • сохраняет все результаты

    Возвращает:
        forecast_df — будущий прогноз (только будущие даты)
        metrics     — словарь {mae, rmse}
    """

    # ============================================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ============================================================
    data_path = os.path.join(BASE, "data/processed/usd_preprocessed.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.rename(columns={"date": "ds", "rate": "y"})

    # Последняя реальная дата
    last_date = df["ds"].max()

    # ============================================================
    # 2. ОБУЧЕНИЕ PROPHET
    # ============================================================
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    model.fit(df)

    # ============================================================
    # 3. BACKTEST — последние 30 дней
    # ============================================================
    df_test = df.tail(90)
    future_test = df_test[["ds"]]

    fc_test = model.predict(future_test)

    y_true = df_test["y"].values
    y_pred = fc_test["yhat"].values

    # Метрики
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {"mae": mae, "rmse": rmse}

    # Сохранение backtest результатов
    backtest_df = pd.DataFrame({
        "date": df_test["ds"],
        "real": y_true,
        "forecast": y_pred
    })

    backtest_path = os.path.join(BASE, "data/processed/prophet_test_predictions.csv")
    os.makedirs(os.path.dirname(backtest_path), exist_ok=True)
    backtest_df.to_csv(backtest_path, index=False)

    # ============================================================
    # 4. ПОСТРОЕНИЕ БУДУЩЕГО ПРОГНОЗА
    # ============================================================
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Оставляем только будущие строки (строго после last_date)
    forecast_future = forecast[forecast["ds"] > last_date]

    forecast_df = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={
            "ds": "date",
            "yhat": "forecast",
            "yhat_lower": "lower",
            "yhat_upper": "upper"
        }
    )

    # Сохранение будущего прогноза
    future_path = os.path.join(BASE, "data/processed/usd_prophet_forecast.csv")
    os.makedirs(os.path.dirname(future_path), exist_ok=True)
    forecast_df.to_csv(future_path, index=False)

    # ============================================================
    # 5. СОХРАНЕНИЕ МЕТРИК
    # ============================================================
    metrics_path = os.path.join(BASE, "models/prophet_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    pd.DataFrame([metrics]).to_json(metrics_path, orient="records")

    print("Prophet готов.")
    print(f"Backtest сохранён: {backtest_path}")
    print(f"Будущий прогноз сохранён: {future_path}")
    print(f"Метрики сохранены: {metrics_path}")

    return forecast_df, metrics
