# src/model/model_lstm.py
"""
LSTM модель для прогнозирования временного ряда USD→UZS.

Основные особенности:
    - Используется классическая архитектура LSTM → (опциональная активация) → Linear.
    - На выход берётся последнее скрытое состояние LSTM.
    - Можно включить ReLU или Tanh для повышения выразительности модели.
    - Реализована усовершенствованная инициализация весов:
        • Weight_ih_*  → Xavier uniform
        • Weight_hh_*  → Orthogonal
        • Bias_*       → zeros
      Это повышает стабильность обучения.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM → (Activation) → Linear регрессионная модель.

    Параметры:
        input_size  — число признаков на шаге времени (по умолчанию 1: только курс).
        hidden_size — размер скрытого состояния LSTM.
        num_layers  — количество вложенных LSTM-слоёв.
        dropout     — dropout между слоями LSTM, используется только если num_layers > 1.
        init_weights — выполнять ли кастомную инициализацию весов.
        activation   — тип активации: "relu", "tanh" или None.

    Пример:
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        y = model(x)  # x: (batch, seq_len, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        init_weights: bool = True,
        activation: Optional[str] = None
    ):
        super().__init__()

        # Если слой один — dropout отключается автоматически
        lstm_dropout = dropout if num_layers > 1 else 0.0

        # --------------------------
        # Основная LSTM-ячейка
        # --------------------------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True  # форматы (batch, seq, features)
        )

        # --------------------------
        # Финальный линейный слой: hidden_size → 1 (регрессия)
        # --------------------------
        self.fc = nn.Linear(hidden_size, 1)

        # --------------------------
        # Опциональная активация
        # --------------------------
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None

        # Сохраняем параметры внутри модели — удобно при сериализации
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # ---------------------------------
        # Кастомная инициализация весов
        # ---------------------------------
        if init_weights:
            self._init_weights()

    # ----------------------------------------------------------- #
    def _init_weights(self):
        """
        Инициализация весов LSTM и Linear слоёв.

        Почему так:
            - Xavier uniform хорошо подходит для входных весов RNN/LSTM.
            - Orthogonal стабилизирует рекуррентные связи.
            - Zero bias снижает случайное смещение на старте обучения.
        """
        for name, param in self.lstm.named_parameters():

            # Входные веса: input → hidden
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            # Рекуррентные веса: hidden → hidden
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            # Все bias
            elif "bias" in name:
                nn.init.zeros_(param)

        # Инициализация весов выходного Linear слоя
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ----------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Прямой проход модели.

        Аргументы:
            x      — входной тензор формы (batch, seq_len, input_size)
            hidden — кортеж (h0, c0). Если None — создаётся автоматически.

        Выход:
            prediction — тензор формы (batch, 1)
        """

        # out — скрытые состояния для каждого шага времени
        # hidden — (h_n, c_n) состояние последнего шага
        out, hidden = self.lstm(x, hidden)

        # Берём скрытое состояние последнего временного шага
        # Формат: (batch, hidden_size)
        last_output = out[:, -1, :]

        # При необходимости — пропускаем через нелинейность
        if self.activation is not None:
            last_output = self.activation(last_output)

        # Пропускаем через линейный слой → итоговый прогноз
        prediction = self.fc(last_output)

        return prediction
