# main.py

"""
Главный управляющий скрипт проекта USD→UZS Forecasting

Поддерживает:
    --fetch          загрузка данных из API
    --preprocess     предобработка данных
    --train-lstm     обучение LSTM
    --predict-lstm   прогноз LSTM
    --prophet        обучение и прогноз Prophet
    --streamlit      запуск UI
    --full           выполнить весь pipeline
"""

import os
import argparse
import subprocess
from datetime import datetime, timedelta

# ============================
# Абсолютные пути
# ============================
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")

PREPROCESS = os.path.join(SRC, "preprocessing", "preprocess.py")
TRAIN_LSTM = os.path.join(SRC, "model", "train.py")
PREDICT_LSTM = os.path.join(SRC, "model", "predict.py")
PROPHET_MODEL = os.path.join(SRC, "model", "prophet_model.py")
STREAMLIT_APP = os.path.join(SRC, "web", "app.py")
FETCH = os.path.join(SRC, "data_loader", "fetch_data.py")


# ============================
# Утилита запуска процессов
# ============================
def run(cmd: str):
    print(f"\n🚀 Запуск: {cmd}")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"❌ Ошибка: {cmd}")
        exit(result.returncode)

    print("✔ Готово!")


# ============================
# Pipeline ФУНКЦИИ
# ============================

def fetch_data(last_days=None):
    if last_days:
        run(f"python {FETCH} --last {last_days}")
    else:
        # по умолчанию 2000 дней загрузки
        end = datetime.now()
        start = end - timedelta(days=2000)
        run(f"python {FETCH} --start {start.date()} --end {end.date()}")


def preprocess():
    run(f"python {PREPROCESS}")


def train_lstm():
    run(f"python {TRAIN_LSTM}")


def predict_lstm():
    run(f"python {PREDICT_LSTM}")


def run_prophet(days=30):
    run(f"python {PROPHET_MODEL} --days {days}" if "--days" in open(PROPHET_MODEL).read()
        else f"python {PROPHET_MODEL}")


def start_streamlit():
    run(f"streamlit run {STREAMLIT_APP}")


# ============================
# MAIN CLI
# ============================
def main():
    parser = argparse.ArgumentParser(description="USD→UZS Forecast Pipeline Manager")

    # Команды
    parser.add_argument("--fetch", action="store_true", help="Загрузить данные из API")
    parser.add_argument("--fetch-last", type=int, help="Загрузить последние N дней")

    parser.add_argument("--preprocess", action="store_true", help="Предобработка данных")

    parser.add_argument("--train-lstm", action="store_true", help="Обучение LSTM модели")
    parser.add_argument("--predict-lstm", action="store_true", help="Создать LSTM прогноз")

    parser.add_argument("--prophet", action="store_true", help="Обучить Prophet и создать прогноз")
    parser.add_argument("--prophet-days", type=int, default=30, help="Горизонт прогноза Prophet")

    parser.add_argument("--streamlit", action="store_true", help="Запуск Streamlit UI")

    parser.add_argument("--full", action="store_true", help="Выполнить весь pipeline")

    args = parser.parse_args()

    # FULL PIPELINE
    if args.full:
        print("\n==================== FULL PIPELINE ====================\n")
        fetch_data(last_days=2000)
        preprocess()
        train_lstm()
        predict_lstm()
        run_prophet(days=args.prophet_days)
        start_streamlit()
        return

    # Индивидуальные команды
    if args.fetch:
        fetch_data()
    if args.fetch_last:
        fetch_data(last_days=args.fetch_last)

    if args.preprocess:
        preprocess()

    if args.train_lstm:
        train_lstm()

    if args.predict_lstm:
        predict_lstm()

    if args.prophet:
        run_prophet(days=args.prophet_days)

    if args.streamlit:
        start_streamlit()


if __name__ == "__main__":
    main()
