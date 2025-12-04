# src/model/dataset.py
"""
Модуль подготовки данных для обучения LSTM модели прогнозирования курса USD→UZS.

Этот файл выполняет несколько ключевых задач:
    1) Загружает предобработанный датасет с диска.
    2) Масштабирует данные с помощью MinMaxScaler.
    3) Разбивает данные на обучающую и тестовую части.
    4) Генерирует временные окна (sequence → target).
    5) Возвращает PyTorch Dataset объекты для дальнейшего обучения.

Формат данных:
    Предобработанный CSV должен содержать столбцы:
        - date        (datetime)
        - rate        (основной таргет)
        - дополнительные признаки (не используются LSTM)

LSTM обучается только на одном признаке — `rate`.
"""

import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

# ============================================================
# Пути
# ============================================================

# Абсолютный путь к корню проекта
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Файл с предобработанными данными
DEFAULT_PREPROCESSED = os.path.join(ROOT, "data/processed/usd_preprocessed.csv")

# ============================================================
# Логирование
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Добавляем handler только один раз (Streamlit может вызывать модуль повторно)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)

# ============================================================
# PyTorch Dataset
# ============================================================

class USDataset(Dataset):
    """
    PyTorch Dataset для временных рядов.

    sequences : numpy.ndarray — форма (samples, window, 1)
    targets   : numpy.ndarray — форма (samples, 1)
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        # Проверка согласованности данных
        assert len(sequences) == len(targets), \
            "Количество входов и выходов должно совпадать"

        # Приводим данные к float32 — формат оптимален для PyTorch
        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        """Количество обучающих примеров."""
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Возвращает один обучающий пример:
            X — окно временного ряда (shape: window_size × 1)
            y — целевое значение (следующий день)
        """
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx])
        )

# ============================================================
# Загрузка предобработанных данных
# ============================================================

def load_preprocessed(path: str = DEFAULT_PREPROCESSED) -> pd.DataFrame:
    """
    Загружает CSV-файл с предобработанными значениями.

    Возвращает:
        DataFrame с колонками:
            date, rate, diff, pct_change, direction, MA7, MA30, ...
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = (
        pd.read_csv(path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Предупреждение, если после обработки доступны пропуски
    if df["rate"].isna().any():
        logger.warning("⚠ Обнаружены пропуски в rate после предобработки.")

    logger.info(f"Предобработанные данные загружены: {len(df)} строк.")
    return df

# ============================================================
# Генерация временных окон
# ============================================================

def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формирует обучающие примеры для LSTM.

    Формат:
        X[i] = data[t : t + window_size]      — окно значений
        y[i] = data[t + window_size]          — следующий день

    Параметры:
        data — одномерный масштабированный массив вида (N, 1)
        window_size — длина временного окна

    Возвращает:
        sequences — X, форма (samples, window_size, 1)
        targets   — y, форма (samples, 1)
    """

    if len(data) <= window_size:
        raise ValueError(
            f"Недостаточно данных ({len(data)}) для окна размером {window_size}"
        )

    sequences = []
    targets = []

    # Скользящее окно
    for i in range(len(data) - window_size):
        seq = data[i : i + window_size]        # окно
        target = data[i + window_size]         # следующее значение
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    logger.info(f"Создано обучающих примеров: {len(sequences)}")
    return sequences, targets

# ============================================================
# Объединённый pipeline подготовки данных
# ============================================================

def prepare_dataset(
    window_size: int = 30,
    test_ratio: float = 0.1,
    data_path: str = DEFAULT_PREPROCESSED
) -> Tuple[Dataset, Dataset, MinMaxScaler]:
    """
    Полный цикл подготовки данных для LSTM.

    Этапы:
        1. Загрузка предобработанного CSV.
        2. Масштабирование значений курса (MinMaxScaler).
        3. Генерация временных окон.
        4. Разделение данных на train/test по времени.
        5. Создание PyTorch Dataset объектов.

    Возвращает:
        train_ds — обучающий набор
        test_ds  — тестовый набор
        scaler   — обученный MinMaxScaler (важен для последующих прогнозов)
    """

    # -------------------------
    # 1. Загружаем DataFrame
    # -------------------------
    df = load_preprocessed(data_path)

    # LSTM использует только столбец 'rate'
    series = df["rate"].values.reshape(-1, 1)

    # -------------------------
    # 2. Масштабируем данные
    # -------------------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    # -------------------------
    # 3. Формируем окна
    # -------------------------
    X, y = create_sequences(scaled, window_size)

    # -------------------------
    # 4. Делим train/test
    # -------------------------
    test_size = max(1, int(len(X) * test_ratio))  # защита: минимум 1 пример

    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test,  y_test  = X[-test_size:], y[-test_size:]

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # -------------------------
    # 5. Создаём Dataset объекты
    # -------------------------
    train_ds = USDataset(X_train, y_train)
    test_ds  = USDataset(X_test, y_test)

    return train_ds, test_ds, scaler
