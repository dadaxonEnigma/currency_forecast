"""
Обучение LSTM модели для прогноза USD→UZS.

Функция train_model выполняет полный ML pipeline:
-------------------------------------------------
1) Загружает данные и нормализует их
2) Создаёт train/test LSTM датасеты
3) Обучает модель и сохраняет лучшую модель
4) Записывает метрики (MAE / RMSE)
5) Сохраняет историю обучения (loss-curve)
6) ДОПОЛНИТЕЛЬНО сохраняет реальные и предсказанные тестовые данные:
       data/processed/lstm_test_predictions.csv

Этот файл отвечает ИСКЛЮЧИТЕЛЬНО за тренировку модели.
"""

import os
import json
import math
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.model.model_lstm import LSTMModel
from src.model.dataset import prepare_dataset


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Добавляем handler один раз
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [TRAIN] %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# ============================================================
# ФИКСАЦИЯ SEED
# ============================================================

def set_seed(seed: int = 42):
    """
    Фиксирует генераторы случайных чисел для воспроизводимости.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# ОБУЧЕНИЕ ОДНОГО ШАГА
# ============================================================

def train_step(model, loader, criterion, optimizer, device):
    """
    Один шаг обучения:
        - вычисляет loss
        - делает backward()
        - обновляет веса
    """
    model.train()
    running_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ============================================================
# ВАЛИДАЦИЯ (EVAL) + СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ
# ============================================================

def eval_step(model, loader, criterion, device):
    """
    Выполняет проход на валидации:
        - считает loss
        - собирает реальные значения и предсказания (в scaled виде)
    """
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)

            running_loss += criterion(preds, y).item()

            # сохраняем в numpy
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    return running_loss / len(loader), np.array(y_true), np.array(y_pred)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ТРЕНИРОВКИ
# ============================================================

def train_model(
    epochs: int = 30,
    batch_size: int = 32,
    window_size: int = 30,
    test_ratio: float = 0.1,
    lr: float = 0.001,
    model_path: str = "models/lstm_usd_model.pth"
):
    """
    Обучает LSTM модель на временном ряду USD→UZS.

    Сохраняет:
        - модель (.pth)
        - scaler.pkl (prepare_dataset делает)
        - models/metrics.json
        - models/loss_curve.csv
        - models/loss_curve.png
        - data/processed/lstm_test_predictions.csv ← важный файл
    """

    set_seed()

    # ------------------------------ #
    # 1) Загружаем DATASET
    # ------------------------------ #
    train_ds, test_ds, scaler = prepare_dataset(window_size, test_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")

    # ------------------------------ #
    # 2) Создаём модель
    # ------------------------------ #
    model = LSTMModel().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    train_losses = []
    test_losses = []

    # ------------------------------ #
    # 3) TRAIN LOOP
    # ------------------------------ #
    for epoch in range(1, epochs + 1):

        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        test_loss, y_true, y_pred = eval_step(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        logger.info(
            f"Epoch {epoch}/{epochs} | Train={train_loss:.6f} | Test={test_loss:.6f}"
        )

        # сохраняем лучшую модель
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"✔ Лучшая модель сохранена (test_loss={best_loss:.6f})")

    # ------------------------------ #
    # 4) ФИНАЛЬНЫЕ МЕТРИКИ
    # ------------------------------ #
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    os.makedirs("models", exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump({"mae": mae, "rmse": rmse}, f, indent=4)

    logger.info(f"Итоговые метрики: MAE={mae:.6f}, RMSE={rmse:.6f}")

    # ------------------------------ #
    # 5) СОХРАНЯЕМ ПРЕДСКАЗАНИЯ TEST-SET
    # ------------------------------ #

    df_raw = pd.read_csv("data/processed/usd_preprocessed.csv", parse_dates=["date"])
    test_len = len(test_ds)

    # реальные даты тестовой выборки
    dates_test = df_raw["date"].iloc[-test_len:].reset_index(drop=True)

    # инверсируем масштаб
    preds_inversed = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    reals_inversed = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    out_df = pd.DataFrame({
        "date": dates_test,
        "real": reals_inversed,
        "lstm_pred": preds_inversed
    })

    os.makedirs("data/processed", exist_ok=True)
    out_df.to_csv("data/processed/lstm_test_predictions.csv", index=False)

    logger.info("✔ Test predictions сохранены → data/processed/lstm_test_predictions.csv")

    # ------------------------------ #
    # 6) СОХРАНЯЕМ КРИВУЮ УБЫТИЙ
    # ------------------------------ #

    loss_df = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_loss": train_losses,
        "test_loss": test_losses
    })

    loss_df.to_csv("models/loss_curve.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
    plt.plot(loss_df["epoch"], loss_df["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/loss_curve.png")
    plt.close()

    logger.info("✔ Loss curve сохранена → models/loss_curve.png")

    return model


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    train_model()
