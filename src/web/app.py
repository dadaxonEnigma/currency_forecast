# src/web/app.py
"""
Streamlit приложение для просмотра исторического курса USD -> UZS.

Запуск:
    streamlit run src/web/app.py
"""
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import os

DATA_PATH = "data/raw/usd_rates.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["diff"] = df["rate"].diff()
    df["direction"] = df["diff"].apply(lambda x: "up" if x > 0 else ("down" if x < 0 else "flat"))
    return df


def main():
    st.title("USD → UZS — Исторический курс (ЦБ РУз)")
    st.markdown("Данные: API Центробанка Узбекистана. Показываются реальные значения курса, "
                "дни роста/падения. (Тестовая версия — прогнозы позже)")

    if not os.path.exists(DATA_PATH):
        st.error(f"CSV не найден по пути {DATA_PATH}. Запустите src/data_loader/fetch_data.py")
        return

    df = load_data(DATA_PATH)

    st.subheader("Обзор данных")
    st.write(df.tail(10))

    st.subheader("График курса")
    fig = px.line(df, x="date", y="rate", title="USD → UZS (реальный курс)")
    # добавим Scatter для точек роста/падения
    up = df[df["direction"] == "up"]
    down = df[df["direction"] == "down"]
    fig.add_scatter(x=up["date"], y=up["rate"], mode="markers", name="Рост", marker=dict(symbol="triangle-up", size=8))
    fig.add_scatter(x=down["date"], y=down["rate"], mode="markers", name="Падение", marker=dict(symbol="triangle-down", size=8))

    st.plotly_chart(fig, use_container_width=True)

    st.sidebar.header("Настройки")
    start = st.sidebar.date_input("Дата начала", df["date"].min().date())
    end = st.sidebar.date_input("Дата конца", df["date"].max().date())
    st.sidebar.write("Показать диапазон:", start, "—", end)

    # Фильтр по датам
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df_filtered = df.loc[mask]
    st.subheader("Счётчик")
    st.metric("Период строк", f"{len(df_filtered)}")

    st.subheader("График (фильтрованный)")
    fig2 = px.line(df_filtered, x="date", y="rate", title="USD → UZS (фильтр)")
    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
