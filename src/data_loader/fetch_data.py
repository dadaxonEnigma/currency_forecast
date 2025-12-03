# src/data_loader/fetch_data.py
"""
Загрузка исторических курсов USD → UZS из API Центрального Банка Узбекистана.

API:
    https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/YYYY-MM-DD/

Каждый запрос возвращает список всех валют за конкретный день.
Нам нужна лишь валюта с "Ccy": "USD".

Данный модуль:
    - генерирует диапазон дат (2018-12-01 → сегодня);
    - делает один запрос на дату;
    - извлекает USD;
    - нормализует дату формата DD.MM.YYYY;
    - формирует DataFrame;
    - сохраняет CSV в data/raw/usd_rates.csv.

Автор: YOU
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


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
# КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────
BASE_URL = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/all"


# ─────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# ОСНОВНАЯ ЛОГИКА ЗАГРУЗКИ
# ─────────────────────────────────────────────────────────────
def fetch_usd_for_date(session: requests.Session, date: datetime) -> Optional[float]:
    """
    Запрашивает данные за конкретную дату и извлекает курс USD.

    Возвращает:
        float (курс) или None, если USD отсутствует.
    """
    formatted = date.strftime("%Y-%m-%d")  # API принимает только YYYY-MM-DD
    url = f"{BASE_URL}/{formatted}/"

    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Ответ {resp.status_code} для {url}")
            return None

        data = resp.json()  # это список валют за день

        for item in data:
            if item.get("Ccy") == "USD":
                rate_str = item.get("Rate")
                try:
                    return float(rate_str.replace(",", ""))
                except Exception:
                    logger.error(f"Не удалось преобразовать Rate '{rate_str}'")
                    return None

        logger.warning(f"USD не найден в данных за {formatted}")
        return None

    except Exception as e:
        logger.error(f"Ошибка при запросе {url}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ ЗАГРУЗКИ
# ─────────────────────────────────────────────────────────────
def fetch_usd_history(
    start_date: str = "2018-12-01",
    end_date: Optional[str] = None,
    out_csv: str = "data/raw/usd_rates.csv",
    sleep_time: float = 0.05,
) -> pd.DataFrame:
    """
    Загружает исторические курсы USD → UZS и сохраняет CSV.

    Возвращает:
        DataFrame с колонками: ['date', 'rate']
    """

    # Преобразование дат
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

    # Создать директорию
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df.to_csv(out_csv, index=False)
    logger.info(f"Сохранено {len(df)} строк в {out_csv}")

    return df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetch_usd_history()
