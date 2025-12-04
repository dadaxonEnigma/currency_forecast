# src/model/prophet_model.py
"""
Модуль обучения и прогнозирования с помощью модели Facebook Prophet.

Prophet — модель аддитивных компонент для временных рядов,
которая автоматически учитывает:
    • тренд
    • годовую сезонность
    • недельную сезонность
и хорошо работает на экономических данных.

Этот модуль:
    - загружает предобработанные данные
    - обучает Prophet на всей истории
    - рассчитывает метрики MAE/RMSE на последнем участке данных
    - делает прогноз на N дней вперёд
    - сохраняет прогноз и метрики в файлы
"""

import os
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Абсолютный путь к директории проекта
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def train_prophet(days: int = 30):
    """
    Обучает модель Prophet и формирует прогноз.

    Параметры:
        days — горизонт прогноза (сколько дней вперёд предсказывать)

    Возвращает:
        out     — DataFrame с будущими прогнозами
        metrics — словарь с ошибками (MAE, RMSE)
    """

    # ============================================================
    # 1. Загрузка данных
    # ============================================================
    data_path = os.path.join(BASE, "data/processed/usd_preprocessed.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден: {data_path}")

    # Prophet требует колонки ds (дата) и y (значение)
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.rename(columns={"date": "ds", "rate": "y"})

    last_date = df["ds"].max()  # последняя реальная дата в истории

    # ============================================================
    # 2. Обучение модели Prophet
    # ============================================================
    model = Prophet(
        yearly_seasonality=True,   # годовая сезонность 365 дней
        weekly_seasonality=True,   # недельная сезонность
        daily_seasonality=False    # дневная сезонность нам не нужна
    )

    model.fit(df)

    # ============================================================
    # 3. Оценка качества модели (тест: последние 30 дней)
    # ============================================================
    df_test = df.tail(30)                # реальный последние 30 дней
    future_test = df_test[["ds"]]        # подаём Prophet даты
    fc_test = model.predict(future_test) # встроенное predict()

    y_true = df_test["y"].values
    y_pred = fc_test["yhat"].values

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {"mae": mae, "rmse": rmse}

    # ============================================================
    # 4. Генерация будущего прогноза
    # ============================================================
    future = model.make_future_dataframe(periods=days)  # реальные + будущие
    forecast = model.predict(future)

    # Оставляем только будущие даты (после last_date)
    forecast = forecast[forecast["ds"] > last_date]

    # Упрощаем формат результата
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
        "ds": "date",
        "yhat": "forecast",
        "yhat_lower": "lower",
        "yhat_upper": "upper"
    })

    # ============================================================
    # 5. Сохранение прогноза
    # ============================================================
    out_path = os.path.join(BASE, "data/processed/usd_prophet_forecast.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    # ============================================================
    # 6. Сохранение метрик
    # ============================================================
    metrics_path = os.path.join(BASE, "models/prophet_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    pd.DataFrame([metrics]).to_json(metrics_path, orient="records")

    print("Prophet готов.")
    print("Прогноз сохранён:", out_path)
    print("Метрики сохранены:", metrics_path)

    return out, metrics
