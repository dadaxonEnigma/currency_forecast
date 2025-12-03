# src/model/train.py
"""
Обучение LSTM модели для прогноза USD→UZS.
Сохраняет:
- lstm_usd_model.pth
- scaler.pkl
- metrics.json (MAE, RMSE)
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import json
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.model.model_lstm import LSTMModel
from src.model.dataset import prepare_dataset


def train_model(
    epochs=30,
    batch_size=32,
    window_size=30,
    test_ratio=0.1,
    lr=0.001,
    save_path="models/lstm_usd_model.pth"
):

    # Загружаем данные
    train_ds, test_ds, scaler = prepare_dataset(window_size, test_ratio)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Девайс
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Модель
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Обучение
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ====================
        # Оценка качества
        # ====================
        model.eval()
        test_loss = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)

                test_loss += criterion(preds, y).item()

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        test_loss /= len(test_loader)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    # ======================
    # Метрики MAE и RMSE
    # ======================
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {"mae": mae, "rmse": rmse}

    os.makedirs("models", exist_ok=True)

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Метрики сохранены:", metrics)

    # ======================
    # Сохранение модели
    # ======================
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")

    # ======================
    # Сохранение scaler
    # ======================
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")

    return model


if __name__ == "__main__":
    train_model()
