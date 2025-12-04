# 💵 Forecasting System
## 📈 Predictive Analytics using LSTM, Prophet & Streamlit

Этот проект — полноценная система для анализа и прогнозирования курса USD → UZS на базе:

- **LSTM (PyTorch)** — глубокое обучение
- **Prophet (Meta)** — статистическое прогнозирование
- **Streamlit** — интерактивный веб-интерфейс

Проект полностью автономен: включает подготовку данных, обучение, прогноз, визуализацию и сравнение моделей.

---

## 🚀 Возможности

### 🤖 Прогнозирование
- LSTM модель с обучением по историческим данным
- Prophet модель для быстрых статистических прогнозов
- Сравнение LSTM vs Prophet

### 📊 Визуализация
- История курса
- KPI валютного ряда
- Прогнозы с зонами роста/падения
- Доверительные интервалы Prophet
- Точки изменения тренда

### 🧱 ML Pipeline
- Предобработка данных
- Заполнение пропусков
- Инженерия признаков
- Масштабирование
- Тренировка и сохранение модели
- Проверка качества и графики

---
## System Architecture
![ML System Architecture](img/full_pipeline(vert).svg)

## Подробные архитектурные диаграммы

#### 1. Data Pipeline - Fetching Data
![Fetching Data Pipeline](img/fetch_data_result.svg)


#### 2. Data Pipeline - Preprocessing
![Data Preprocessing Pipeline](img/preprocessing.svg)

#### 3. Dataset Pipeline - Windows / Scaler / Split
![Dataset Preparation Pipeline](img/min_max.svg)

#### 4. Training Pipeline - LSTM Training Loop
![LSTM Training Pipeline](img/train_graph.svg)

#### 5. Prophet Model Pipeline
![Prophet Forecast Pipeline](img/prophet.svg)

## 🧱 Структура проекта
📦 project
│
├── 📁 src
│   ├── 📁 preprocessing
│   │   └── 🧹 preprocess.py                  # очистка данных, заполнение пропусков
│   │
│   ├── 📁 model
│   │   ├── 🧩 dataset.py                     # windows, scaler, split
│   │   ├── 🧠 model_lstm.py                  # архитектура LSTM
│   │   ├── 🔮 predict.py                     # LSTM прогноз (inference)
│   │   └── 🎯 train.py                       # обучение LSTM + метрики
│   │
│   └── 📁 web
│       └── 🖥️ app.py                         # Streamlit интерфейс
│
├── 📁 data
│   ├── 📁 raw
│   │   └── 📄 usd_rates.csv                  # исходные данные
│   │
│   └── 📁 processed
│       ├── 📄 usd_preprocessed.csv           # очищенные данные
│       ├── 📄 usd_forecast.csv               # прогноз LSTM
│       ├── 📄 usd_prophet_forecast.csv       # прогноз Prophet
│       └── 📄 lstm_test_predictions.csv      # предсказания на тесте
│
├── 📁 models
│   ├── 🔧 lstm_usd_model.pth                 # веса модели
│   ├── 🔧 scaler.pkl                         # MinMaxScaler
│   ├── ⚙️ model_config.json                  # параметры окна и модели
│   ├── 📊 loss_curve.png                     # график обучения
│   ├── 📄 loss_curve.csv                     # данные кривой обучения
│   └── 📄 metrics.json                       # MAE/RMSE
│
├── 📝 pyproject.toml
└── 📝 README.md

---

## 🔧 Установка

### 1. Клонируйте проект
```bash
git clone git@github.com:dadaxonEnigma/currency_forecast.git
cd currency_forecast
```
### 2. Установите зависимости
```bash
pip install .
```

## 📥 1. Предобработка данных
Для начала нужно создать обработанный датасет:

```bash
python src/preprocessing/preprocess.py
```
После выполнения появится:
data/processed/usd_preprocessed.csv

### 📊 Пример визуализации
![alt text](img/rate_data.png)
![alt text](img/preprocessing.png)

## 🤖 2. Обучение LSTM модели
Запустите тренировку:

```bash
python src/model/train.py
```
### 📉 Loss Curve (пример)
![alt text](img/learning_curve.png)

## 🔮 3. Генерация прогноза
### Прогноз LSTM:
```bash
python src/model/predict.py
```
### Прогноз Prophet:
```bash
python -c "from src.model.prophet_model import train_prophet; train_prophet(days=30)"
```

## 🖥 4. Запуск Streamlit UI
```bash
streamlit run src/web/app.py
```
Приложение включает вкладки:
* 📘 История

* 📈 LSTM прогноз

* 🔮 Prophet прогноз

* ⚔️ Сравнение моделей

* 📊 Пример визуализаций

### История + прогноз LSTM
![alt text](img/pred_lstm.png)

### Прогноз Prophet
![alt text](img/pred_prophet.png)
![alt text](img/pred_prophet2.png)

### Сравнение моделей
![alt text](img/compare_models.png)

## 🤝 Автор
Dadakhon Turgunboev
Machine Learning Engineer
GitHub: [https://github.com/yourprofile](https://github.com/dadaxonEnigma)
