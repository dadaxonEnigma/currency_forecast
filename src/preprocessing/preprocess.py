# src/preprocessing/preprocess.py
"""
Предобработка данных USD → UZS временного ряда.

Функции:
    - load_raw_data()        – загрузка сырых данных
    - build_full_date_range() – создание непрерывного ряда дат
    - preprocess_usd_data()   – основная очистка и генерация признаков
    - save_preprocessed()     – сохранение результата

Метод заполнения пропусков:
    B — forward-fill (как в банковской практике)

Выход:
    data/processed/usd_preprocessed.csv
"""

import os
import logging
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# ЛОГИРОВАНИЕ
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# ─────────────────────────────────────────────────────────────
# ЗАГРУЗКА СЫРЫХ ДАННЫХ
# ─────────────────────────────────────────────────────────────
def load_raw_data(path: str = "data/raw/usd_rates.csv") -> pd.DataFrame:
    """Загружает сырой датасет."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"Загружено {len(df)} строк из {path}")
    return df


# ─────────────────────────────────────────────────────────────
# СОЗДАНИЕ НЕПРЕРЫВНОГО РЯДА ДАТ
# ─────────────────────────────────────────────────────────────
def build_full_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт полный календарный ряд между min и max датами."""
    start = df["date"].min()
    end = df["date"].max()

    full_range = pd.DataFrame({"date": pd.date_range(start, end)})
    merged = full_range.merge(df, on="date", how="left")

    logger.info("Создан непрерывный ряд дат.")
    return merged


# ─────────────────────────────────────────────────────────────
# ОСНОВНАЯ ПРЕДОБРАБОТКА
# ─────────────────────────────────────────────────────────────
def preprocess_usd_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка:
        ✓ непрерывный ряд дат
        ✓ forward-fill пропусков
        ✓ генерация признаков (diff, pct_change, MA7, MA30)
        ✓ классификация direction
    """

    df = build_full_date_range(df)

    # Forward-fill пропусков
    df["rate"] = df["rate"].fillna(method="ffill")

    # Если первые значения были NaN (до появления первого реального)
    df["rate"] = df["rate"].fillna(method="bfill")

    # diff
    df["diff"] = df["rate"].diff()

    # pct_change
    df["pct_change"] = df["rate"].pct_change()

    # direction
    df["direction"] = df["diff"].apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
    )

    # Скользящие средние
    df["MA7"] = df["rate"].rolling(7).mean()
    df["MA30"] = df["rate"].rolling(30).mean()

    logger.info("Предобработка завершена.")
    return df


# ─────────────────────────────────────────────────────────────
# СОХРАНЕНИЕ
# ─────────────────────────────────────────────────────────────
def save_preprocessed(df: pd.DataFrame, path: str = "data/processed/usd_preprocessed.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Предобработанный датасет сохранён в {path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = preprocess_usd_data(df_raw)
    save_preprocessed(df_clean)
