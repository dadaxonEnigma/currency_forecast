# src/web/app.py
"""
Streamlit приложение для USD→UZS:
- Raw data
- Processed data
- KPI
- LSTM прогноз
- Сравнение прогноз vs реальность
"""

import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

# =======================
# Добавляем ROOT в sys.path
# =======================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.model.predict import predict_future


# ===========================
# Функции загрузки данных
# ===========================
@st.cache_data
def load_raw():
    path = "data/raw/usd_rates.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date")


@st.cache_data
def load_processed():
    path = "data/processed/usd_preprocessed.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date")


def main():
    st.title("📈 USD → UZS Аналитика и Прогноз LSTM")

    df_raw = load_raw()
    df_proc = load_processed()

    # Ошибка если данных нет
    if df_raw is None:
        st.error("Нет файла data/raw/usd_rates.csv. Сначала выполните fetch_data.py")
        return

    # =======================================================
    # KPI БЛОК
    # =======================================================

    st.header("📊 KPI валютного курса")

    if df_proc is not None:
        latest = df_proc.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        col1.metric("Текущий курс", f"{latest['rate']:,.2f}")

        col2.metric("MA7", f"{latest['MA7']:,.2f}" if pd.notna(latest["MA7"]) else "—")

        col3.metric("MA30", f"{latest['MA30']:,.2f}" if pd.notna(latest["MA30"]) else "—")

        col4.metric(
            "Изменение (diff)",
            f"{latest['diff']:+.2f}" if pd.notna(latest["diff"]) else "—"
        )

        col5.metric(
            "Изменение (%)",
            f"{latest['pct_change'] * 100:+.3f}%" if pd.notna(latest["pct_change"]) else "—"
        )

    else:
        st.warning("Нет обработанных данных — запустите preprocess.py")


    # =======================================================
    # Вкладки
    # =======================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📘 Raw Data",
        "🧹 Processed Data",
        "📈 Прогноз курса",
        "📊 Сравнение прогноза"
    ])

    # -------------------------------------------------------
    # TAB 1 — RAW DATA
    # -------------------------------------------------------
    with tab1:
        st.header("📘 Исходные данные")
        st.dataframe(df_raw.tail(10))

        fig_raw = px.line(df_raw, x="date", y="rate", title="Raw USD→UZS")
        st.plotly_chart(fig_raw, use_container_width=True)

    # -------------------------------------------------------
    # TAB 2 — PROCESSED
    # -------------------------------------------------------
    with tab2:
        if df_proc is None:
            st.warning("Нет обработанных данных.")
        else:
            st.header("🧹 Обработанные данные")
            st.dataframe(df_proc.tail(10))

            fig_proc = px.line(df_proc, x="date", y="rate", title="Processed USD→UZS")
            st.plotly_chart(fig_proc, use_container_width=True)

    # -------------------------------------------------------
    # TAB 3 — FORECAST
    # -------------------------------------------------------
    with tab3:
        st.header("📈 Прогноз курса USD→UZS")

        days = st.slider("Горизонт прогноза (дни)", 7, 60, 14)

        if st.button("Сделать прогноз"):
            st.info("Выполняется прогноз...")

            try:
                fc = predict_future(days=days)
                st.success("Прогноз успешно выполнен!")

                st.subheader("Таблица прогноза")
                st.dataframe(fc)

                df_plot = load_raw()
                merged = pd.concat([df_plot, fc], ignore_index=True)

                fig_fc = px.line(
                    merged,
                    x="date",
                    y=["rate", "forecast"],
                    title="Прогноз USD→UZS"
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                # Загружаем метрики модели
                try:
                    import json
                    with open("models/metrics.json", "r") as f:
                        metrics = json.load(f)

                    col1, col2 = st.columns(2)
                    col1.metric("MAE модели", f"{metrics['mae']:.4f}")
                    col2.metric("RMSE модели", f"{metrics['rmse']:.4f}")

                except:
                    st.warning("Метрики модели не найдены.")

            except Exception as e:
                st.error(f"Ошибка прогноза: {e}")

    # -------------------------------------------------------
    # TAB 4 — COMPARE FORECAST
    # -------------------------------------------------------
    with tab4:
        st.header("📊 Сравнение реального курса и прогноза")

        df_raw = load_raw()

        forecast_path = "data/processed/usd_forecast.csv"
        if not os.path.exists(forecast_path):
            st.warning("Сначала выполните прогноз.")
        else:
            df_fc = pd.read_csv(forecast_path, parse_dates=["date"])

            history_end = df_raw["date"].max()
            forecast_start = df_fc["date"].min()

            st.write(f"Последняя дата в истории: {history_end}")
            st.write(f"Первая дата прогноза: {forecast_start}")

            fig = px.line(title="Исторический курс vs Прогноз")

            fig.add_scatter(
                x=df_raw["date"],
                y=df_raw["rate"],
                mode="lines",
                name="История",
                line=dict(color="blue")
            )

            fig.add_scatter(
                x=df_fc["date"],
                y=df_fc["forecast"],
                mode="lines+markers",
                name="Прогноз",
                line=dict(color="green")
            )

            # -----------------------------
            # Вертикальная линия без ошибок
            # -----------------------------
            fig.add_shape(
                type="line",
                x0=history_end,
                y0=df_raw["rate"].min(),
                x1=history_end,
                y1=df_raw["rate"].max(),
                line=dict(color="gray", width=2, dash="dash")
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📘 Таблица прогноза")
            st.dataframe(df_fc.tail(10))





if __name__ == "__main__":
    main()
