# src/preprocessing/preprocess.py
"""
Предобработка временного ряда USD → UZS.

Этапы обработки:
    1. Загрузка сырых данных.
    2. Создание полного календаря (без пропусков по датам).
    3. Заполнение отсутствующих значений (финансовый forward-fill).
    4. Генерация инженерных признаков:
        • diff        — абсолютное изменение курса
        • pct_change  — процентное изменение
        • direction   — тренд ("up", "down", "flat")
        • MA7 / MA30  — скользящие средние

Результат сохраняется в:
    data/processed/usd_preprocessed.csv

Этот модуль используется перед обучением моделей LSTM и Prophet.
"""

import os
import logging
import pandas as pd

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Чтобы Streamlit/повторный импорт не дублировал логи
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)


# ============================================================
# ЗАГРУЗКА СЫРЫХ ДАННЫХ
# ============================================================

def load_raw_data(path: str = "data/raw/usd_rates.csv") -> pd.DataFrame:
    """
    Загружает CSV-файл с историческими значениями курса.

    Возвращает DataFrame со столбцами:
        • date
        • rate

    Особенности:
        - удаляет дубликаты дат
        - сортирует даты по возрастанию
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    # Защита: API иногда присылает повторяющиеся даты
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    logger.info(f"Загружено {len(df)} строк из файла {path}")
    return df


# ============================================================
# СОЗДАНИЕ ПОЛНОГО КАЛЕНДАРЯ
# ============================================================

def ensure_full_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт непрерывную последовательность дат от min(date) до max(date).

    Это важно, так как:
        - API может возвращать пропуски по праздникам/выходным,
        - модели временных рядов ожидают регулярный интервал.

    В пропущенные дни значения rate будут заполнены позже.
    """
    start, end = df["date"].min(), df["date"].max()

    # Формируем календарь: один день — одна строка
    full = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})

    # Левый джойн: сохраняем существующие значения, добавляем пустые строки
    df_full = full.merge(df, on="date", how="left")

    logger.info(f"Сформирован календарь: {start.date()} → {end.date()}")
    return df_full


# ============================================================
# ЗАПОЛНЕНИЕ ПРОПУСКОВ В КУРСАХ
# ============================================================

def fill_missing_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в столбце 'rate'.

    Для финансовых временных рядов используется:
        • forward-fill — перенос последнего известного значения
        • backward-fill — для пропусков в начале ряда

    Такой подход обеспечивает корректность данных для моделей.
    """
    missing_before = df["rate"].isna().sum()

    df["rate"] = df["rate"].fillna(method="ffill")   # основной способ
    df["rate"] = df["rate"].fillna(method="bfill")   # защита для первого дня

    missing_after = df["rate"].isna().sum()
    filled = missing_before - missing_after

    logger.info(f"Заполнено пропусков: {filled}")
    return df


# ============================================================
# ИНЖЕНЕРИЯ ПРИЗНАКОВ
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует дополнительные признаки, используемые в анализе и моделях.

    Признаки:
        diff        — абсолютное изменение курса
        pct_change  — процентное изменение курса
        direction   — текстовый тренд ("up", "down", "flat")
        MA7 / MA30  — 7-дневные и 30-дневные скользящие средние
    """

    df["diff"] = df["rate"].diff()
    df["pct_change"] = df["rate"].pct_change()

    # Логическое направление изменения
    df["direction"] = df["diff"].apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
    )

    # Скользящие средние как индикаторы трендов
    df["MA7"] = df["rate"].rolling(7).mean()
    df["MA30"] = df["rate"].rolling(30).mean()

    logger.info("Добавлены признаки: diff, pct_change, direction, MA7, MA30")
    return df


# ============================================================
# ПОЛНЫЙ PIPELINE ПРЕДОБРАБОТКИ
# ============================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полный pipeline предобработки данных:

        1. Создание полного календаря
        2. Заполнение пропусков
        3. Генерация признаков

    Возвращает готовый датасет для обучения моделей.
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
    Сохраняет итоговый предобработанный датасет.

    Создаёт директорию, если её нет.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    logger.info(f"Файл сохранён: {path} ({len(df)} строк)")


# ============================================================
# CLI — запуск из терминала
# ============================================================

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = preprocess(df_raw)
    save_preprocessed(df_clean)
