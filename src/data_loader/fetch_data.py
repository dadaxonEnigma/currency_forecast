# src/data_loader/fetch_data.py
"""
Загрузка исторических курсов USD → UZS из API ЦБ Республики Узбекистан.

Что сделано:
- Упрощён импорт и структура (убраны дубли и sys.path модификации).
- Оставлен CLI-интерфейс (совместимость с текущим workflow).
- Защищённый HTTP session с ретраями и backoff.
- Плавная загрузка с прогресс-баром и опцией паузы между запросами.

Примеры:
    # скачать последние 2000 дней
    python src/data_loader/fetch_data.py --last 2000

    # скачать за период
    python src/data_loader/fetch_data.py --start 2018-12-01 --end 2025-12-03
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------------
# Константы / пути
# ------------------------
# Корень проекта — вычисляется относительно текущего файла
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Файл, куда сохраняются сырые данные (usd_rates.csv)
RAW_OUT = os.path.join(ROOT, "data", "raw", "usd_rates.csv")

# ------------------------
# Логирование
# ------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)

# ------------------------
# API-константы
# ------------------------
BASE_URL = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/all"
USD_CODE = "USD"        # код валюты в ответе API
FIELD_RATE = "Rate"     # поле со значением курса (строка с десятичным разделителем запятыми)
FIELD_CCY = "Ccy"       # поле с кодом валюты

# ------------------------
# HTTP session с retry
# ------------------------
def create_session(retries: int = 5, backoff_factor: float = 0.3) -> requests.Session:
    """
    Создаёт requests.Session с политикой повторных попыток.
    Это нужно, чтобы устойчиво обрабатывать временные ошибки сети и 5xx ответы.
    """
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ------------------------
# Генератор дат (включительно)
# ------------------------
def date_range(start: datetime, end: datetime):
    """Генератор дат с шагом 1 день (включительно end)."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

# ------------------------
# Получение курса за одну дату
# ------------------------
def fetch_usd_rate_for_date(session: requests.Session, date: datetime, timeout: int = 10) -> Optional[float]:
    """
    Запрашивает архивную страницу для конкретной даты и извлекает курс USD.
    Возвращает float курс или None, если курс не найден/ошибка.
    """
    formatted = date.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{formatted}/"

    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            logger.warning("[%s] API status: %s", formatted, resp.status_code)
            return None

        data = resp.json()
        # В ответе — список валют; ищем элемент с полем Ccy == 'USD'
        for item in data:
            if item.get(FIELD_CCY) == USD_CODE:
                # Rate в виде строки "12345.67" или "12,345.67" — убираем запятые
                rate_str = item.get(FIELD_RATE)
                if rate_str is None:
                    return None
                # Заменяем запятые на пустоту, затем приводим к float
                return float(str(rate_str).replace(",", ""))
        logger.warning("[%s] USD not found in API response.", formatted)
        return None

    except Exception as exc:
        logger.error("[%s] Error fetching rate: %s", formatted, exc)
        return None

# ------------------------
# Основная функция: скачивание диапазона дат
# ------------------------
def download_usd_rates(start: datetime, end: datetime, sleep: float = 0.05) -> pd.DataFrame:
    """
    Загружает курсы USD по всем датам в интервале [start, end].
    Параметры:
        start, end: datetime
        sleep: пауза между запросами в секундах (для уменьшения нагрузки на API)
    Возвращает:
        DataFrame с колонками ['date', 'rate'] (date — datetime.date)
    """
    session = create_session()
    records = []

    logger.info("Downloading USD rates: %s → %s", start.date(), end.date())

    # tqdm показывает прогресс по числу дней в диапазоне
    for dt in tqdm(date_range(start, end), desc="Downloading USD"):
        rate = fetch_usd_rate_for_date(session, dt)
        if rate is not None:
            records.append({"date": dt.date(), "rate": rate})
        if sleep:
            # Используем локальную задержку, чтобы не спамить API
            import time
            time.sleep(sleep)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    logger.info("Finished download — rows: %d", len(df))
    return df

# ------------------------
# Сохранение DataFrame в CSV
# ------------------------
def save_dataframe(df: pd.DataFrame, path: str = RAW_OUT):
    """
    Сохраняет DataFrame в CSV, создавая директорию при необходимости.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)

# ------------------------
# CLI — запуск из командной строки
# ------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch USD→UZS historical rates from CBU API")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--last", type=int, help="Download last N days (overrides --start)")
    parser.add_argument("--out", type=str, default=RAW_OUT, help="Output CSV path")

    args = parser.parse_args()

    # Вычисление интервала дат (по умолчанию — с 2018-12-01 до сегодня)
    if args.last:
        end = datetime.now()
        start = end - timedelta(days=args.last)
    else:
        start = datetime.strptime(args.start or "2018-12-01", "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()

    df = download_usd_rates(start, end)
    save_dataframe(df, args.out)
