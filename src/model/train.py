# src/model/train.py
"""
Тренировка LSTM модели для прогноза USD → UZS.

Полный ML pipeline включает:
    1) загрузку и нормализацию данных
    2) подготовку обучающего и тестового датасета
    3) обучение LSTM модели и выбор наилучшей версии
    4) сохранение итоговых метрик MAE/RMSE
    5) построение loss-кривой
    6) сохранение реальных и предсказанных значений тестовой выборки

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
# Логирование
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [TRAIN] %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# ============================================================
# Фиксация SEED — важнейшая часть reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """
    Устанавливает фиксированный seed для всех источников случайности.
    Это делает обучение воспроизводимым.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Один шаг обучения
# ============================================================

def train_step(model, loader, criterion, optimizer, device):
    """
    Выполняет один полный проход по обучающему датасету.

    Этапы:
        - прямой проход модели
        - вычисление функции потерь
        - обратное распространение ошибки
        - обновление весов оптимизатором
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
# Валидация (без обновления весов)
# ============================================================

def eval_step(model, loader, criterion, device):
    """
    Выполняет проход по тестовой выборке:
        - вычисляет среднюю функцию потерь
        - сохраняет реальные и предсказанные значения (масштабированные)

    Возвращает:
        avg_loss — средняя ошибка
        y_true   — реальные значения
        y_pred   — предсказания модели
    """
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():  # отключаем вычисление градиентов
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)

            running_loss += criterion(preds, y).item()

            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    return running_loss / len(loader), np.array(y_true), np.array(y_pred)


# ============================================================
# Основная функция обучения
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
    Запускает полный цикл обучения модели LSTM.

    Сохраняет:
        • веса модели (.pth)
        • scaler.pkl (создаётся prepare_dataset)
        • метрики обучения (models/metrics.json)
        • график и CSV кривой обучения
        • предсказания тестовой выборки
    """

    # 1. Фиксируем seed
    set_seed()

    # -------------------------------------------------------------
    # 1) Готовим датасет (загрузка CSV → масштабирование → окна)
    # -------------------------------------------------------------
    train_ds, test_ds, scaler = prepare_dataset(window_size, test_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")

    # -------------------------------------------------------------
    # 2) Создаём модель и инструменты обучения
    # -------------------------------------------------------------
    model = LSTMModel().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    train_losses = []
    test_losses = []

    # -------------------------------------------------------------
    # 3) Основной цикл тренировки
    # -------------------------------------------------------------
    for epoch in range(1, epochs + 1):

        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        test_loss, y_true, y_pred = eval_step(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        logger.info(f"Epoch {epoch}/{epochs} | Train={train_loss:.6f} | Test={test_loss:.6f}")

        # Сохраняем лучшую модель по качеству на тесте
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"✔ Лучшая модель сохранена (test_loss={best_loss:.6f})")

    # -------------------------------------------------------------
    # 4) Финальные метрики
    # -------------------------------------------------------------
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    os.makedirs("models", exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump({"mae": mae, "rmse": rmse}, f, indent=4)

    logger.info(f"Итоговые метрики: MAE={mae:.6f}, RMSE={rmse:.6f}")

    # -------------------------------------------------------------
    # 5) Сохранение тестовых предсказаний (real vs predicted)
    # -------------------------------------------------------------
    df_raw = pd.read_csv("data/processed/usd_preprocessed.csv", parse_dates=["date"])
    test_len = len(test_ds)

    dates_test = df_raw["date"].iloc[-test_len:].reset_index(drop=True)

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

    # -------------------------------------------------------------
    # 6) Построение и сохранение кривой обучения
    # -------------------------------------------------------------
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
# CLI запуск
# ============================================================

if __name__ == "__main__":
    train_model()
