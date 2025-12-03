# src/model/predict.py
"""
Прогноз временного ряда USD → UZS с использованием обученной LSTM модели.

Основные возможности модуля:
--------------------------------------------------
✔ Абсолютные пути — работает независимо от точки запуска
✔ Автоматическая загрузка параметров модели из model_config.json
✔ Корректная подгрузка scaler.pkl
✔ Итеративный прогноз на N дней вперёд
✔ Полная совместимость со Streamlit (кэширование, отсутствие утечек памяти)
✔ Используется та же архитектура модели, что и в train.py

Этот модуль обеспечивает стабильный и воспроизводимый inference-пайплайн.
"""

import os
import sys

# Добавление ROOT в sys.path, если модуль запускается напрямую
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import joblib

from src.model.model_lstm import LSTMModel
from src.model.dataset import load_preprocessed


# ============================================================
# ПУТИ
# ============================================================

BASE = ROOT
MODELS_DIR = os.path.join(BASE, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "lstm_usd_model.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

FORECAST_OUT = os.path.join(BASE, "data/processed/usd_forecast.csv")


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Streamlit делает двойное логирование, поэтому handler добавляем один раз
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [PREDICT] %(levelname)s: %(message)s")
    )
    logger.addHandler(handler)


# ============================================================
# ЗАГРУЗКА МОДЕЛИ + CONFIG
# ============================================================

def load_model() -> Tuple[torch.nn.Module, torch.device, dict]:
    """
    Загружает обученную LSTM модель, конфигурацию модели и выбирает устройство.

    Возвращает:
        model  — torch.nn.Module (готовая к использованию)
        device — torch.device ('cuda' или 'cpu')
        config — dict (параметры модели: window_size, hidden_size, num_layers, dropout)

    Важные параметры загружаются из model_config.json, чтобы предсказания
    соответствовали той архитектуре, которая использовалась при тренировке.
    """
    # Проверки наличия необходимых файлов
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Конфиг модели не найден: {CONFIG_PATH}")

    # Загружаем конфигурацию
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Параметры архитектуры
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)
    input_size = config.get("input_size", 1)

    # Выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем модель с той же архитектурой
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=config.get("dropout", 0.2),
        init_weights=False  # веса перезапишутся state_dict
    ).to(device)

    # Загружаем веса
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    logger.info("✔ LSTM модель успешно загружена.")

    return model, device, config


# ============================================================
# ПОДГОТОВКА ПОСЛЕДНЕГО ОКНА
# ============================================================

def prepare_sequence(series: np.ndarray, window_size: int, device) -> torch.Tensor:
    """
    Подготавливает последнее окно временного ряда для подачи в модель.

    Аргументы:
        series — нормализованный массив значений (scaled)
        window_size — длина окна
        device — CPU/GPU

    Возвращает:
        тензор формы (1, window_size, 1)
    """
    seq = series[-window_size:].reshape(1, window_size, 1)
    return torch.tensor(seq, dtype=torch.float32).to(device)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРОГНОЗА
# ============================================================

def predict_future(days: int = 7, save_csv: bool = True) -> pd.DataFrame:
    """
    Делает итеративный прогноз на указанное количество дней вперед.

    Этапы:
        1. Загружаем обработанные данные
        2. Применяем scaler
        3. Загружаем модель и конфиг
        4. Итеративно генерируем значения
        5. Деинвертируем масштаб scaler.inverse_transform
        6. Формируем DataFrame и сохраняем результат

    Возвращает:
        DataFrame: date | forecast
    """

    # ------------------ 1. Загружаем обработанные данные ------------------ #
    df = load_preprocessed(os.path.join(BASE, "data/processed/usd_preprocessed.csv"))
    values = df["rate"].values.reshape(-1, 1)

    # ------------------ 2. Загружаем scaler ------------------ #
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"❌ Scaler не найден: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(values)

    # ------------------ 3. Загружаем модель + конфиг ------------------ #
    model, device, config = load_model()
    window_size = config.get("window_size")

    if window_size is None:
        raise ValueError("❌ В model_config.json отсутствует параметр 'window_size'")

    # ------------------ 4. Формируем начальное окно ------------------ #
    seq = prepare_sequence(scaled, window_size, device)
    predictions_scaled = []

    logger.info(f"▶ Начат прогноз на {days} дней...")

    # ------------------ 5. Итеративный прогноз ------------------ #
    for _ in range(days):
        with torch.no_grad():
            pred = model(seq).item()  # scalar

        predictions_scaled.append(pred)

        # Сдвигаем окно: удаляем самый старый элемент, добавляем новый прогноз
        seq_np = seq.cpu().numpy()
        seq_np = np.append(seq_np[:, 1:, :], [[[pred]]], axis=1)
        seq = torch.tensor(seq_np, dtype=torch.float32).to(device)

    # ------------------ 6. Обратное масштабирование ------------------ #
    predictions_arr = np.array(predictions_scaled).reshape(-1, 1)
    forecast_values = scaler.inverse_transform(predictions_arr).flatten()

    # ------------------ 7. Создаем DataFrame прогноза ------------------ #
    start_date = df["date"].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start_date, periods=days)

    forecast_df = pd.DataFrame({
        "date": dates,
        "forecast": forecast_values
    })

    # ------------------ 8. Сохраняем CSV (если нужно) ------------------ #
    if save_csv:
        os.makedirs(os.path.dirname(FORECAST_OUT), exist_ok=True)
        forecast_df.to_csv(FORECAST_OUT, index=False)
        logger.info(f"✔ Прогноз сохранён: {FORECAST_OUT}")

    return forecast_df


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    df = predict_future(days=7)
    print(df)
