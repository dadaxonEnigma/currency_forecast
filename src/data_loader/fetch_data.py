# src/data_loader/fetch_data.py

"""
Загрузка исторических курсов USD → UZS из API ЦБ Узбекистана.
Полностью совместимо с проектной структурой.

Исправления:
- абсолютные пути
- интеграция с main.py
- безопасная CLI логика
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Generator

import pandas as pd
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================================
# Абсолютный путь к проекту
# ==============================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_OUT = os.path.join(ROOT, "data/raw/usd_rates.csv")

# ==============================================
# Логирование
# ==============================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)

# ==============================================
# API CONST
# ==============================================
BASE_URL = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/all"
USD_CODE = "USD"
FIELD_RATE = "Rate"
FIELD_CCY = "Ccy"

# ==============================================
# Создание HTTP session
# ==============================================
def create_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ==============================================
# Генератор дат
# ==============================================
def date_range(start: datetime, end: datetime):
    while start <= end:
        yield start
        start += timedelta(days=1)

# ==============================================
# Загрузка курса за одну дату
# ==============================================
def fetch_usd_rate_for_date(session: requests.Session, date: datetime) -> Optional[float]:
    formatted = date.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{formatted}/"

    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"[{formatted}] API status: {response.status_code}")
            return None

        data = response.json()
        for item in data:
            if item.get(FIELD_CCY) == USD_CODE:
                return float(item.get(FIELD_RATE).replace(",", ""))

        logger.warning(f"[{formatted}] USD not found in API response.")
        return None

    except Exception as e:
        logger.error(f"[{formatted}] Error: {e}")
        return None

# ==============================================
# Основная функция загрузки
# ==============================================
def download_usd_rates(start: datetime, end: datetime, sleep: float = 0.05) -> pd.DataFrame:
    session = create_session()
    records = []

    logger.info(f"Downloading USD rates: {start.date()} → {end.date()}")

    for d in tqdm(date_range(start, end), desc="Downloading USD"):
        rate = fetch_usd_rate_for_date(session, d)
        if rate is not None:
            records.append({"date": d.date(), "rate": rate})
        if sleep:
            import time
            time.sleep(sleep)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df

# ==============================================
# Сохранение файла
# ==============================================
def save_dataframe(df: pd.DataFrame, path: str = RAW_OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows → {path}")

# ==============================================
# CLI
# ==============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch USD→UZS history")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--last", type=int, help="Download last N days")
    parser.add_argument("--out", type=str, default=RAW_OUT)

    args = parser.parse_args()

    # Последние N дней
    if args.last:
        end = datetime.now()
        start = end - timedelta(days=args.last)
    else:
        start = datetime.strptime(args.start or "2018-12-01", "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()

    df = download_usd_rates(start, end)
    save_dataframe(df, args.out)
