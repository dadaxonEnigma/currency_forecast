# src/model/predict.py
"""
Использует обученную модель для прогноза курса USD→UZS.
"""

import torch
import numpy as np
import pandas as pd
from src.model.model_lstm import LSTMModel
from src.model.dataset import load_preprocessed
import joblib


def predict_future(days=7, window_size=30, model_path="models/lstm_usd_model.pth"):

    df = load_preprocessed()
    values = df["rate"].values.reshape(-1, 1)

    # загрузка scaler
    scaler = joblib.load("models/scaler.pkl")
    scaled = scaler.transform(values)

    # загрузка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # последние окна
    last_seq = scaled[-window_size:].reshape(1, window_size, 1)
    last_seq = torch.tensor(last_seq, dtype=torch.float32).to(device)

    predictions = []

    for _ in range(days):
        with torch.no_grad():
            pred = model(last_seq)
        predictions.append(pred.item())

        # обновляем окно (shift)
        new_value = pred.item()
        last_seq = np.append(last_seq.cpu().numpy()[:, 1:, :], [[[new_value]]], axis=1)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).to(device)

    # обратная трансформация
    predictions = np.array(predictions).reshape(-1, 1)
    predicted_rates = scaler.inverse_transform(predictions).flatten()

    # создаём таблицу
    future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": predicted_rates
    })

    forecast_df.to_csv("data/processed/usd_forecast.csv", index=False)
    print("Прогноз сохранён:", "data/processed/usd_forecast.csv")

    return forecast_df


if __name__ == "__main__":
    predict_future(days=7)
