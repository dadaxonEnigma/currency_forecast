# src/preprocessing/preprocess.py
"""
Модуль предобработки временного ряда USD → UZS.

Этапы обработки данных:
1. Загрузка сырых данных.
2. Создание полного календаря без пропусков дат.
3. Заполнение отсутствующих значений (типично для финансовых рядов).
4. Генерация инженерных признаков:
    - разница (diff)
    - процентное изменение
    - направление движения
    - скользящие средние MA7 / MA30
5. Сохранение итогового набора данных.

Результат:
    data/processed/usd_preprocessed.csv
"""

import os
import logging
import pandas as pd
from typing import Optional


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Добавить handler только 1 раз (Streamlit иначе дублирует логи)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# ============================================================
# ЗАГРУЗКА СЫРЫХ ДАННЫХ
# ============================================================

def load_raw_data(path: str = "data/raw/usd_rates.csv") -> pd.DataFrame:
    """
    Загружает CSV-файл с курсами валют.

    Параметры:
        path (str): путь к файлу данных

    Возвращает:
        pd.DataFrame: таблица с колонками date, rate, ...

    Исключения:
        FileNotFoundError — если файл не найден.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    # Удаляем дубликаты дат, так как финансовые API иногда присылают повторения
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    logger.info(f"Загружено {len(df)} строк из файла {path}")
    return df


# ============================================================
# СОЗДАНИЕ ПОЛНОГО КАЛЕНДАРЯ
# ============================================================

def ensure_full_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит непрерывный календарный ряд, чтобы не было пропусков по датам.

    Например:
        было: 2020-01-01, 2020-01-03
        станет: 2020-01-01, 2020-01-02, 2020-01-03

    Параметры:
        df (pd.DataFrame): исходный датасет

    Возвращает:
        pd.DataFrame: датасет с полной последовательностью дат
    """
    start, end = df["date"].min(), df["date"].max()

    # Формируем полный календарь по дням
    full = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})

    # Делаем left-join, чтобы сохранить существующие значения и добавить пустые
    df_full = full.merge(df, on="date", how="left")

    logger.info(f"Сформирован полный календарь: {start.date()} → {end.date()}")
    return df_full


# ============================================================
# ЗАПОЛНЕНИЕ ПРОПУСКОВ
# ============================================================

def fill_missing_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет отсутствующие значения курса.

    Финансовые ряды обычно используют forward fill:
        - выходные
        - праздники
        - пропуски в API

    Метод:
        ffill — копирует последнее известное значение.
        bfill — подстраховка для начала ряда.

    Возвращает:
        pd.DataFrame: датасет без NaN в rate
    """
    missing_before = df["rate"].isna().sum()

    # Forward fill — основа
    df["rate"] = df["rate"].fillna(method="ffill")
    # Backward fill — если пропуски были в начале
    df["rate"] = df["rate"].fillna(method="bfill")

    missing_after = df["rate"].isna().sum()
    filled = missing_before - missing_after

    logger.info(f"Заполнено пропусков: {filled}")
    return df


# ============================================================
# ИНЖЕНЕРИЯ ПРИЗНАКОВ
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт основные признаки, полезные для анализа и моделей:

    - diff: абсолютное изменение курса
    - pct_change: процентное изменение
    - direction: текстовая метка "up", "down", "flat"
    - MA7 / MA30: скользящие средние

    Возвращает:
        pd.DataFrame: датасет с дополнительными фичами
    """

    # Абсолютное и относительное изменение
    df["diff"] = df["rate"].diff()
    df["pct_change"] = df["rate"].pct_change()

    # Тренд в виде категориального признака
    df["direction"] = df["diff"].apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
    )

    # Скользящие средние — классика анализа трендов
    df["MA7"] = df["rate"].rolling(7).mean()
    df["MA30"] = df["rate"].rolling(30).mean()

    logger.info("Добавлены признаки: diff, pct_change, direction, MA7, MA30")
    return df


# ============================================================
# ПОЛНЫЙ ЦИКЛ ПРЕДОБРАБОТКИ
# ============================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Основная функция предобработки данных.
    На вход принимает сырой датасет, на выходе возвращает очищенный и
    полностью подготовленный DataFrame.

    Этапы:
    1. ensure_full_calendar()
    2. fill_missing_rates()
    3. add_features()
    """
    df = ensure_full_calendar(df)
    df = fill_missing_rates(df)
    df = add_features(df)
    return df


# ============================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ============================================================

def save_preprocessed(df: pd.DataFrame, path: str = "data/processed/usd_preprocessed.csv") -> None:
    """
    Сохраняет готовый предобработанный DataFrame в CSV.

    Параметры:
        df (pd.DataFrame): итоговый датасет
        path (str): путь файла назначения
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Файл сохранён: {path} ({len(df)} строк)")


# ============================================================
# CLI — Позволяет запускать модуль напрямую
# ============================================================

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = preprocess(df_raw)
    save_preprocessed(df_clean)
