# src/model/dataset.py
"""
Подготовка данных для LSTM (PyTorch):
- загрузка предобработанных данных
- MinMax нормализация
- создание окон (window_size)
- разделение train/test
- PyTorch Dataset для DataLoader
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


class USDataset(Dataset):
    """PyTorch Dataset для оконного временного ряда."""

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


def load_preprocessed(path="data/processed/usd_preprocessed.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError("Файл не найден: data/processed/usd_preprocessed.csv")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def create_sequences(data, window_size=30):
    """
    Создаёт выборки вида:
        X = последних 30 дней
        y = следующий день
    """
    sequences = []
    targets = []

    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        target = data[i + window_size]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def prepare_dataset(window_size=30, test_ratio=0.1):
    df = load_preprocessed()

    # Используем только колонку rate → временной ряд
    values = df["rate"].values.reshape(-1, 1)

    # Масштабируем
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Оконные выборки
    X, y = create_sequences(scaled, window_size)

    # Train / Test split
    test_size = int(len(X) * test_ratio)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    return (
        USDataset(X_train, y_train),
        USDataset(X_test, y_test),
        scaler
    )
