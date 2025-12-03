import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
#  Абсолютные пути к корню проекта и данным
#  Позволяет запускать приложение из любой директории
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ML-модули
from src.model.predict import predict_future
from src.model.prophet_model import train_prophet

# Пути к данным
RAW_PATH = os.path.join(ROOT, "data/raw/usd_rates.csv")
PROC_PATH = os.path.join(ROOT, "data/processed/usd_preprocessed.csv")
LSTM_FC_PATH = os.path.join(ROOT, "data/processed/usd_forecast.csv")
PROPHET_FC_PATH = os.path.join(ROOT, "data/processed/usd_prophet_forecast.csv")


# ============================================================
#  КЭШИРОВАНИЕ ДАННЫХ
# ============================================================

@st.cache_data
def load_raw():
    """
    Загружает сырые данные USD→UZS.
    Используется кэширование Streamlit для ускорения рендера UI.
    """
    if os.path.exists(RAW_PATH):
        return pd.read_csv(RAW_PATH, parse_dates=["date"]).sort_values("date")
    return None


@st.cache_data
def load_processed():
    """
    Загружает предобработанные данные.
    """
    if os.path.exists(PROC_PATH):
        return pd.read_csv(PROC_PATH, parse_dates=["date"]).sort_values("date")
    return None


def clear_cache():
    """Полностью очищает кэш приложения Streamlit."""
    st.cache_data.clear()


# ============================================================
#  KPI — Показатели текущего состояния валютного рынка
# ============================================================

def render_kpi(df_proc: pd.DataFrame):
    """
    Рендер KPI карточек:
    - текущий курс
    - изменение за сутки
    - MA7 и MA30 (скользящие средние)
    """
    st.header("📊 KPI валютного курса")

    if df_proc is None:
        st.warning("Нет обработанных данных.")
        return

    last = df_proc.iloc[-1]
    prev = df_proc.iloc[-2]

    # Суточная динамика
    change = last["rate"] - prev["rate"]
    change_pct = (change / prev["rate"]) * 100

    arrow = "🟢↑" if change > 0 else "🔴↓" if change < 0 else "➡"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Текущий курс", f"{last['rate']:.2f}")
    col2.metric("Суточное изменение", f"{change:+.2f}", f"{arrow} {change_pct:+.2f}%")
    col3.metric("MA7", f"{last['MA7']:.2f}")
    col4.metric("MA30", f"{last['MA30']:.2f}")


# ============================================================
#  TAB 1 — Исторические данные
# ============================================================

def render_raw_tab(df_raw):
    """
    Рисует таблицу последних значений и график истории курсов.
    """
    st.subheader("📘 Исторические данные USD→UZS")
    st.dataframe(df_raw.tail(20))

    fig = px.line(df_raw, x="date", y="rate", title="История курса USD→UZS")
    fig.update_traces(line_color="royalblue")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  TAB 2 — Прогноз LSTM
# ============================================================

def render_lstm_tab():
    """
    Генерирует и отображает прогноз LSTM:
    - стрелка роста/падения
    - точки роста/падения на графике
    - красивая визуализация в стиле Prophet
    """
    st.subheader("📈 Улучшенный LSTM прогноз USD→UZS")

    days = st.slider("Горизонт прогноза (дни)", 7, 120, 30)

    if st.button("Сделать LSTM прогноз"):
        st.info("Генерация прогноза...")

        df_pred = predict_future(days=days)
        clear_cache()
        st.success("Прогноз готов!")

        df_raw = load_raw()

        # ------------------------- Изменение курса ------------------------- #
        diff = df_pred["forecast"].iloc[-1] - df_raw["rate"].iloc[-1]
        pct = (diff / df_raw["rate"].iloc[-1]) * 100
        arrow = "🟢↑" if diff > 0 else "🔴↓" if diff < 0 else "➡"

        st.metric(
            "Изменение относительно последнего курса",
            f"{diff:+.2f}",
            f"{arrow} {pct:+.2f}%"
        )

        # ------------------------- Создание маркеров ------------------------- #
        df_pred_plot = df_pred.copy()
        df_pred_plot["diff"] = df_pred_plot["forecast"].diff()

        df_pred_plot["color"] = df_pred_plot["diff"].apply(
            lambda x: "green" if x > 0 else ("red" if x < 0 else "gray")
        )

        df_pred_plot["arrow"] = df_pred_plot["diff"].apply(
            lambda x: "▲" if x > 0 else ("▼" if x < 0 else "•")
        )

        # ------------------------- ГРАФИК ------------------------- #
        fig = go.Figure()

        # История
        fig.add_trace(go.Scatter(
            x=df_raw["date"], y=df_raw["rate"],
            mode="lines", line=dict(color="#2c3e50", width=2.5), name="История"
        ))

        # Прогноз LSTM
        fig.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["forecast"],
            mode="lines", line=dict(color="#00a86b", width=3),
            name="Прогноз LSTM"
        ))

        # "Зона прогноза" (условная мягкая область)
        fig.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["forecast"],
            mode="lines", line=dict(width=0), showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=df_pred["date"],
            y=[df_pred["forecast"].min()] * len(df_pred),
            fill="tonexty",
            fillcolor="rgba(0,168,107,0.15)",
            line=dict(width=0),
            name="Зона прогноза",
            hoverinfo="skip"
        ))

        # Маркеры роста/падения
        fig.add_trace(go.Scatter(
            x=df_pred_plot["date"],
            y=df_pred_plot["forecast"],
            mode="markers+text",
            marker=dict(size=9, color=df_pred_plot["color"], line=dict(width=1, color="black")),
            text=df_pred_plot["arrow"],
            textposition="top center",
            name="Рост / Падение"
        ))

        fig.update_layout(
            title="📈 История + Прогноз LSTM",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_pred.tail(10))


# ============================================================
#  TAB 3 — Прогноз Prophet
# ============================================================

def render_prophet_tab():
    """
    Генерация прогноза Prophet + красивая визуализация:
    - доверительный интервал
    - стрелки изменения
    - маркеры роста/падения
    """
    st.subheader("🔮 Улучшенный прогноз Prophet")

    days = st.slider("Горизонт Prophet (дни)", 7, 180, 30)

    if st.button("Запустить Prophet"):
        st.info("Prophet обучается...")

        df_fc, metrics = train_prophet(days=days)
        clear_cache()

        st.success("Прогноз готов!")

        df_raw = load_raw()
        df_proc = load_processed()

        # ------------------------- Изменение курса ------------------------- #
        last_real = df_proc.iloc[-1]["rate"]
        diff = df_fc["forecast"].iloc[-1] - last_real
        pct = (diff / last_real) * 100
        arrow = "🟢↑" if diff > 0 else "🔴↓" if diff < 0 else "➡"

        st.metric("Изменение (Prophet)", f"{diff:+.2f}", f"{arrow} {pct:+.2f}%")

        # ------------------------- Маркеры Prophet ------------------------- #
        df_fc_plot = df_fc.copy()
        df_fc_plot["diff"] = df_fc_plot["forecast"].diff()

        df_fc_plot["color"] = df_fc_plot["diff"].apply(
            lambda x: "green" if x > 0 else ("red" if x < 0 else "gray")
        )
        df_fc_plot["arrow"] = df_fc_plot["diff"].apply(
            lambda x: "▲" if x > 0 else ("▼" if x < 0 else "•")
        )

        # ------------------------- График Prophet ------------------------- #
        fig = go.Figure()

        # История
        fig.add_trace(go.Scatter(
            x=df_raw["date"], y=df_raw["rate"],
            mode="lines", line=dict(color="#2c3e50", width=2.5),
            name="История"
        ))

        # Основная линия прогноза
        fig.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["forecast"],
            mode="lines", line=dict(color="#0057b7", width=3),
            name="Прогноз Prophet"
        ))

        # Доверительный интервал
        fig.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["upper"],
            mode="lines", line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=df_fc["date"], y=df_fc["lower"],
            fill="tonexty",
            fillcolor="rgba(0, 113, 227, 0.15)",
            line=dict(width=0),
            name="Доверительный интервал"
        ))

        # Маркеры
        fig.add_trace(go.Scatter(
            x=df_fc_plot["date"], y=df_fc_plot["forecast"],
            mode="markers+text",
            marker=dict(size=9, color=df_fc_plot["color"], line=dict(width=1, color="black")),
            text=df_fc_plot["arrow"],
            textposition="top center",
            name="Рост / Падение"
        ))

        fig.update_layout(
            title="🔮 История + Прогноз Prophet",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Метрики Prophet
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{metrics['mae']:.4f}")
        col2.metric("RMSE", f"{metrics['rmse']:.4f}")

        st.dataframe(df_fc.tail(10))


# ============================================================
#  TAB 4 — Сравнение моделей
# ============================================================

def render_model_compare():
    """
    Сравнивает прогнозы LSTM и Prophet на одном графике
    + маркеры роста/падения у обеих моделей.
    """
    st.subheader("⚔️ Сравнение моделей LSTM и Prophet")

    df_raw = load_raw()

    # Проверяем наличие данных
    if not os.path.exists(LSTM_FC_PATH):
        st.warning("Сначала выполните LSTM прогноз.")
        return

    if not os.path.exists(PROPHET_FC_PATH):
        st.warning("Сначала выполните Prophet прогноз.")
        return

    df_lstm = pd.read_csv(LSTM_FC_PATH, parse_dates=["date"])
    df_prophet = pd.read_csv(PROPHET_FC_PATH, parse_dates=["date"])

    # Функция для генерации маркеров
    def make_markers(df, column):
        df = df.copy()
        df["diff"] = df[column].diff()
        df["color"] = df["diff"].apply(
            lambda x: "green" if x > 0 else ("red" if x < 0 else "gray")
        )
        df["arrow"] = df["diff"].apply(
            lambda x: "▲" if x > 0 else ("▼" if x < 0 else "•")
        )
        return df

    df_lstm_m = make_markers(df_lstm, "forecast")
    df_prophet_m = make_markers(df_prophet, "forecast")

    # ------------------------- ГРАФИК ------------------------- #
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_raw["date"], y=df_raw["rate"],
        mode="lines", line=dict(color="#2c3e50", width=2),
        name="История"
    ))

    fig.add_trace(go.Scatter(
        x=df_lstm["date"], y=df_lstm["forecast"],
        mode="lines", line=dict(color="#00a86b", width=3),
        name="LSTM"
    ))

    fig.add_trace(go.Scatter(
        x=df_lstm_m["date"], y=df_lstm_m["forecast"],
        mode="markers",
        marker=dict(size=8, color=df_lstm_m["color"], line=dict(width=1, color="black")),
        name="LSTM точки"
    ))

    fig.add_trace(go.Scatter(
        x=df_prophet["date"], y=df_prophet["forecast"],
        mode="lines", line=dict(color="#0057b7", width=3),
        name="Prophet"
    ))

    fig.add_trace(go.Scatter(
        x=df_prophet_m["date"], y=df_prophet_m["forecast"],
        mode="markers",
        marker=dict(size=8, color=df_prophet_m["color"], line=dict(width=1, color="black")),
        name="Prophet точки"
    ))

    fig.update_layout(
        title="⚔️ Сравнение моделей: LSTM vs Prophet",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Таблица сравнения
    st.write("📋 Последние значения прогнозов")
    combo = df_lstm.merge(df_prophet, on="date", how="inner", suffixes=("_LSTM", "_Prophet"))
    st.dataframe(combo.tail(20))


# ============================================================
#  MAIN — Точка входа в приложение Streamlit
# ============================================================

def main():
    """
    Главная функция: создаёт интерфейс Streamlit,
    загружает данные и отображает вкладки.
    """
    st.set_page_config(page_title="USD→UZS Analytics", layout="wide")
    st.title("💵 USD → UZS Аналитика и прогноз")

    df_raw = load_raw()
    df_proc = load_processed()

    if df_raw is None:
        st.error("Нет данных. Сначала загрузите данные.")
        return

    if df_proc is not None:
        render_kpi(df_proc)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📘 История",
        "📈 LSTM прогноз",
        "🔮 Prophet прогноз",
        "⚔️ Сравнение моделей"
    ])

    with tab1:
        render_raw_tab(df_raw)

    with tab2:
        render_lstm_tab()

    with tab3:
        render_prophet_tab()

    with tab4:
        render_model_compare()


if __name__ == "__main__":
    main()
