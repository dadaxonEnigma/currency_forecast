# src/data_loader/fetch_data.py
"""
Загрузка исторических курсов USD → UZS из API Центрального Банка Узбекистана.

API (по одному дню):
    https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/YYYY-MM-DD/

Функционал:
    ✓ Загрузка полного датасета (2018-12-01 → сегодня)
    ✓ Загрузка за произвольный диапазон дат
    ✓ Загрузка последних N дней (--last 30 / 90 / 7)
    ✓ Автосоздание директорий
    ✓ Логирование
    ✓ Повторные попытки при сетевых ошибках

Выход:
    data/raw/usd_rates.csv
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
# НАСТРОЙКИ
# ─────────────────────────────────────────────────────────────
BASE_URL = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/all"


def create_session() -> requests.Session:
    """Создаёт requests.Session с retry-политикой."""
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


def date_range(start: datetime, end: datetime):
    """Генератор дат включительно."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_usd_for_date(session: requests.Session, date: datetime) -> Optional[float]:
    """
    Получает курс USD → UZS за конкретный день.
    Возвращает float или None.
    """
    formatted = date.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{formatted}/"

    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"[{formatted}] Ответ API: {resp.status_code}")
            return None

        data = resp.json()  # это список всех валют за день

        for item in data:
            if item.get("Ccy") == "USD":
                rate_str = item.get("Rate")
                return float(rate_str.replace(",", ""))

        logger.warning(f"[{formatted}] USD не найден в данных")
        return None

    except Exception as e:
        logger.error(f"Ошибка запроса за {formatted}: {e}")
        return None


def fetch_usd_history(
    start_date: str = "2018-12-01",
    end_date: Optional[str] = None,
    out_csv: str = "data/raw/usd_rates.csv",
    sleep_time: float = 0.05,
) -> pd.DataFrame:
    """
    Главная функция загрузки.

    Args:
        start_date: str, формат YYYY-MM-DD
        end_date: str или None
        out_csv: путь для сохранения итогового CSV
        sleep_time: задержка между запросами, чтобы не спамить API

    Returns:
        pandas.DataFrame с колонками ['date', 'rate']
    """

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    session = create_session()
    records = []

    logger.info(f"Загрузка USD курсов: {start.date()} → {end.date()}")

    for d in tqdm(list(date_range(start, end)), desc="Downloading USD history"):
        rate = fetch_usd_for_date(session, d)

        if rate is not None:
            records.append({"date": d.date(), "rate": rate})
        else:
            logger.warning(f"Пропуск даты: {d.date()}")

        time.sleep(sleep_time)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    logger.info(f"Сохранено {len(df)} строк в {out_csv}")

    return df


# ─────────────────────────────────────────────────────────────
# CLI интерфейс
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download USD historical exchange rates")

    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--last", type=int, help="Download only last N days")

    args = parser.parse_args()

    # РЕЖИМ: только последние N дней
    if args.last:
        end = datetime.now()
        start = end - timedelta(days=args.last)

        fetch_usd_history(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )

    else:
        # РЕЖИМ: диапазон дат или полный период
        fetch_usd_history(
            start_date=args.start or "2018-12-01",
            end_date=args.end,
        )
