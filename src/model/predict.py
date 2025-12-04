# src/model/predict.py
"""
Модуль прогнозирования временного ряда USD → UZS с использованием обученной LSTM модели.

Что делает:
    • Загружает модель, scaler и конфигурацию архитектуры.
    • Формирует последнее окно данных и делает итеративный прогноз.
    • Деинвертирует (inverse_transform) масштабирование.
    • Сохраняет прогноз в CSV (используется Streamlit UI).
    • Полностью независим от рабочей директории благодаря абсолютным путям.

Этот файл обеспечивает стабильный и воспроизводимый inference-процесс.
"""

import os
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
# Пути проекта
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = ROOT

MODELS_DIR = os.path.join(BASE, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "lstm_usd_model.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

FORECAST_OUT = os.path.join(BASE, "data/processed/usd_forecast.csv")


# ============================================================
# Логирование
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [PREDICT] %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# ============================================================
# Загрузка модели и конфигурации
# ============================================================

def load_model() -> Tuple[torch.nn.Module, torch.device, dict]:
    """
    Загружает:
        - обученную модель LSTM,
        - конфигурацию архитектуры (window_size, hidden_size, num_layers, dropout),
        - устройство (CPU/GPU).

    Возвращает:
        model  — torch.nn.Module
        device — torch.device
        config — dict
    """

    # Проверяем наличие всех необходимых файлов
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Конфиг модели не найден: {CONFIG_PATH}")

    # Загружаем конфиг
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Читаем параметры архитектуры
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)
    input_size = config.get("input_size", 1)

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаём модель с теми же параметрами, что и при обучении
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=config.get("dropout", 0.2),
        init_weights=False  # Веса будут загружены из файла
    ).to(device)

    # Загружаем веса модели
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # режим инференса

    logger.info("✔ LSTM модель успешно загружена")

    return model, device, config


# ============================================================
# Формирование последнего окна данных
# ============================================================

def prepare_sequence(series: np.ndarray, window_size: int, device) -> torch.Tensor:
    """
    Формирует тензор последнего окна временного ряда.

    Вход:
        series — масштабированные значения (N × 1)
        window_size — длина окна
        device — CPU/GPU

    Выход:
        тензор формы (1, window_size, 1)
    """

    seq = series[-window_size:].reshape(1, window_size, 1)
    return torch.tensor(seq, dtype=torch.float32).to(device)


# ============================================================
# Основная функция прогноза
# ============================================================

def predict_future(days: int = 7, save_csv: bool = True) -> pd.DataFrame:
    """
    Итеративный прогноз LSTM на указанное количество дней вперёд.

    Этапы:
        1. Загружает предобработанные данные.
        2. Применяет обученный scaler.
        3. Загружает модель и конфиг.
        4. Делаем прогноз на каждый следующий день последовательно.
        5. Выполняет inverse_transform.
        6. Формирует финальный DataFrame.

    Возвращает:
        DataFrame с колонками: date | forecast
    """

    # ------------------------- 1. Загружаем данные -------------------------
    df = load_preprocessed(os.path.join(BASE, "data/processed/usd_preprocessed.csv"))
    values = df["rate"].values.reshape(-1, 1)

    # ------------------------- 2. Загружаем scaler -------------------------
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"❌ Scaler не найден: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(values)

    # ------------------------- 3. Загружаем модель и конфиг -------------------------
    model, device, config = load_model()
    window_size = config.get("window_size")

    if window_size is None:
        raise ValueError("❌ В model_config.json отсутствует параметр 'window_size'")

    # ------------------------- 4. Формируем начальное окно -------------------------
    seq = prepare_sequence(scaled, window_size, device)
    predictions_scaled = []

    logger.info(f"▶ Генерация прогноза на {days} дней...")

    # ------------------------- 5. Итеративное предсказание -------------------------
    for _ in range(days):
        with torch.no_grad():
            pred = model(seq).item()  # получаем scalar float

        predictions_scaled.append(pred)

        # Сдвигаем окно: удаляем старый элемент и добавляем новый прогноз
        seq_np = seq.cpu().numpy()
        seq_np = np.append(seq_np[:, 1:, :], [[[pred]]], axis=1)

        seq = torch.tensor(seq_np, dtype=torch.float32).to(device)

    # ------------------------- 6. Обратное масштабирование -------------------------
    predictions_arr = np.array(predictions_scaled).reshape(-1, 1)
    forecast_values = scaler.inverse_transform(predictions_arr).flatten()

    # ------------------------- 7. Формируем DataFrame -------------------------
    start_date = df["date"].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start_date, periods=days)

    forecast_df = pd.DataFrame({
        "date": dates,
        "forecast": forecast_values
    })

    # ------------------------- 8. Сохраняем результат -------------------------
    if save_csv:
        os.makedirs(os.path.dirname(FORECAST_OUT), exist_ok=True)
        forecast_df.to_csv(FORECAST_OUT, index=False)
        logger.info(f"✔ Прогноз сохранён → {FORECAST_OUT}")

    return forecast_df


# ============================================================
# CLI (ручной запуск)
# ============================================================

if __name__ == "__main__":
    df = predict_future(days=7)
    print(df)
