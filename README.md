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

## 🧱 Структура проекта
📦 project
│
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── model/
│   │   ├── dataset.py
│   │   ├── model_lstm.py
│   │   ├── predict.py
│   │   └── train.py
│   └── web/
│       └── app.py     ← Streamlit приложение
│
├── data/
│   ├── raw/
│   │   └── usd_rates.csv
│   └── processed/
│       ├── usd_preprocessed.csv
│       ├── usd_forecast.csv
│       ├── usd_prophet_forecast.csv
│       └── lstm_test_predictions.csv
│
├── models/
│   ├── lstm_usd_model.pth
│   ├── scaler.pkl
│   ├── model_config.json
│   ├── loss_curve.png
│   ├── loss_curve.csv
│   └── metrics.json
│
├── pyproject.toml
└── README.md

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
![alt text](../itm/img/image-1.png)
![alt text](../itm/img/image.png)

## 🤖 2. Обучение LSTM модели
Запустите тренировку:

```bash
python src/model/train.py
```
Будут созданы:

models/lstm_usd_model.pth

models/scaler.pkl

models/model_config.json

models/metrics.json

models/loss_curve.png

data/processed/lstm_test_predictions.csv

#### 📉 Loss Curve (пример)
https://example.com/loss_curve.png

## 🔮 3. Генерация прогноза
Прогноз LSTM:

```bash
python src/model/predict.py
```
Результат:
data/processed/usd_forecast.csv

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

История + прогноз LSTM
https://example.com/lstm_forecast.png

Прогноз Prophet
https://example.com/prophet_forecast.png

Сравнение моделей
https://example.com/comparison.png

### 🧪 Тестовые предсказания
После тренировки система автоматически сохраняет:

Формат:

date	real	lstm_pred
2024-05-01	12700	12695
2024-05-02	12705	12710
🧠 Архитектура LSTM
text
Input (window_size)
        ↓
      LSTM layers
        ↓
 Optional Activation
        ↓
       Linear
        ↓
     Output (forecast)

🤝 Автор
Dadakhon Turgunboev
Machine Learning Engineer
GitHub: [https://github.com/yourprofile](https://github.com/dadaxonEnigma)
