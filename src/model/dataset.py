# src/model/dataset.py
"""
Подготовка данных для обучения LSTM модели.

Функции модуля обеспечивают:
    ✔ Загрузку предобработанных данных с абсолютным путём.
    ✔ Генерацию временных окон для модели.
    ✔ Масштабирование данных с помощью MinMaxScaler.
    ✔ Создание PyTorch Dataset объектов для Train / Test.
    ✔ Гарантированную работу даже с маленькими датасетами.
"""

import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


# ============================================================
# ПУТИ К ПРОЕКТУ
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE = ROOT
DEFAULT_PREPROCESSED = os.path.join(BASE, "data/processed/usd_preprocessed.csv")


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Добавляем handler один раз, иначе Streamlit дублирует вывод
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# ============================================================
# DATASET CLASS
# ============================================================

class USDataset(Dataset):
    """
    Класс PyTorch Dataset для временных рядов.

    Параметры:
        sequences — numpy массив формы (samples, window_size, 1)
        targets   — numpy массив формы (samples, 1)
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        assert len(sequences) == len(targets), \
            "Количество входов и выходов должно совпадать"

        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        """Количество обучающих примеров."""
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """Возвращает одну пару (X, y) в формате PyTorch tensors."""
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx])
        )


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_preprocessed(path: str = DEFAULT_PREPROCESSED) -> pd.DataFrame:
    """
    Загружает предобработанный CSV-файл.

    Параметры:
        path — абсолютный путь к файлу

    Возвращает:
        pd.DataFrame с колонками [date, rate, diff, pct_change, ...]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    if df["rate"].isna().any():
        logger.warning("⚠ Обнаружены пропуски в rate после предобработки.")

    logger.info(f"Предобработанные данные загружены ({len(df)} строк).")
    return df


# ============================================================
# ГЕНЕРАЦИЯ ОКОН ВРЕМЕННОГО РЯДА
# ============================================================

def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует обучающие примеры для LSTM.

    Формат:
        X[i] = data[t : t + window_size]
        y[i] = data[t + window_size]

    Параметры:
        data — одномерный нормализованный массив (scaled)
        window_size — длина окна

    Возвращает:
        sequences — массив X формы (samples, window_size, 1)
        targets   — массив y формы (samples, 1)
    """

    if len(data) <= window_size:
        raise ValueError(
            f"Слишком мало данных ({len(data)}) для окна window_size={window_size}"
        )

    sequences, targets = [], []

    # Окно "скользит" по ряду
    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]      # N значений
        target = data[i + window_size]     # Следующее значение
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    logger.info(f"Создано {len(sequences)} обучающих примеров.")
    return sequences, targets


# ============================================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ TRAIN / TEST
# ============================================================

def prepare_dataset(
    window_size: int = 30,
    test_ratio: float = 0.1,
    data_path: str = DEFAULT_PREPROCESSED
) -> Tuple[Dataset, Dataset, MinMaxScaler]:
    """
    Полная подготовка данных для обучения модели LSTM.

    Параметры:
        window_size — длина окна временного ряда
        test_ratio — доля тестовых данных (0.1 = 10%)
        data_path — путь к предобработанному CSV

    Возвращает:
        train_ds  — PyTorch Dataset (train)
        test_ds   — PyTorch Dataset (test)
        scaler    — обученный MinMaxScaler
    """

    # 1. Загружаем данные
    df = load_preprocessed(data_path)
    series = df["rate"].values.reshape(-1, 1)

    # 2. Масштабируем (обязательно сохраняем scaler!)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    # 3. Генерируем окна
    X, y = create_sequences(scaled, window_size)

    # 4. Хронологическое разделение
    test_size = max(1, int(len(X) * test_ratio))  # гарантируем ≥ 1 точку

    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 5. Превращаем в PyTorch Dataset
    train_ds = USDataset(X_train, y_train)
    test_ds = USDataset(X_test, y_test)

    return train_ds, test_ds, scaler
