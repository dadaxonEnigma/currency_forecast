# src/model/model_lstm.py
"""
Улучшенная LSTM модель для регрессии временного ряда (USD → UZS).

Ключевые идеи и пояснения:
- Используем nn.LSTM (batch_first=True) — удобно для DataLoader с (batch, seq, feat).
- На выходе берём последнее скрытое состояние (последний временной шаг) и
  прогоняем через Linear для получения скалярного прогноза.
- Добавлена опциональная нелинейность перед линейным слоем (ReLU / Tanh) —
  иногда полезно для стабилизации или усиления выразительности модели.
- Инициализация весов: Xavier для входных весов, Orthogonal для рекуррентных,
  нули для смещений. Это хорошая практика для RNN/LSTM.
- Параметры сохранены в экземпляре (hidden_size, num_layers, dropout) чтобы
  downstream code (train/predict) мог к ним обратиться при необходимости.

Интерфейс модели не изменён:
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    preds = model(x)  # x shape: (batch, seq_len, input_size)
"""

import os
import sys

# Удобный root path — позволяет импортировать проект при запуске модуля напрямую
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Tuple, Optional
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM -> (Activation?) -> Linear модель для регрессии временного ряда.

    Параметры конструктора:
        input_size (int): число признаков на шаге времени (обычно 1 для курса).
        hidden_size (int): размер скрытого состояния LSTM.
        num_layers (int): количество вложенных LSTM-слоёв.
        dropout (float): dropout между слоями LSTM (работает только если num_layers>1).
        init_weights (bool): если True — выполняется кастомная инициализация весов.
        activation (Optional[str]): "relu", "tanh" или None — нелинейность перед Linear.

    Почему так:
        - batch_first=True удобнее для dataloader'ов (batch, seq, feat).
        - Инициализация весов помогает стабильному и более быстрому обучению.
        - Опциональная активация даёт лёгкую гибкость без усложнения архитектуры.
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

        # Если num_layers == 1, dropout для LSTM игнорируется — делаем 0.0 явно
        lstm_dropout = dropout if num_layers > 1 else 0.0

        # LSTM: возвращает всю последовательность output + скрытые состояния (h_n, c_n)
        # out shape -> (batch, seq_len, hidden_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )

        # Финальный линейный слой, преобразует hidden_size -> 1 (регрессия)
        self.fc = nn.Linear(hidden_size, 1)

        # Опциональная активация между LSTM и Linear
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None

        # Сохраняем параметры в объекте — удобно для сериализации/логирования
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Выполняем кастомную инициализацию весов, если нужно
        if init_weights:
            self._init_weights()

    # ----------------------------------------------------------- #
    def _init_weights(self):
        """
        Кастомная инициализация весов для LSTM и Linear.

        Подход:
            - weight_ih_* (входные веса): Xavier uniform — хорошая начальная инициализация.
            - weight_hh_* (рекуррентные веса): Orthogonal — часто рекомендуют для RNN.
            - bias: zeros.

        Примечание:
            PyTorch по умолчанию уже использует разумные инициализации, но явная
            инициализация помогает воспроизводимости и экспериментам.
        """
        for name, param in self.lstm.named_parameters():
            # weight_ih_l[k] — веса для входа -> hidden (input-hidden)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            # weight_hh_l[k] — рекуррентные веса (hidden-hidden)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            # bias_ih_l[k], bias_hh_l[k]
            elif "bias" in name:
                nn.init.zeros_(param)

        # Инициализация линейного слоя
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
            x (torch.Tensor): входной тензор формы (batch, seq_len, input_size).
            hidden (Optional[Tuple[Tensor, Tensor]]): начальные (h0, c0) состояния LSTM,
                каждый тензор формы (num_layers, batch, hidden_size). Если None — PyTorch
                инициализирует нулями.

        Возвращает:
            prediction (torch.Tensor): прогноз формы (batch, 1)
        """

        # out: все скрытые состояния для каждого шага времени
        # hidden: кортеж (h_n, c_n) — состояния последнего шага
        out, hidden = self.lstm(x, hidden)

        # Берём скрытое состояние последнего временного шага (последний элемент sequence dim)
        # out[:, -1, :] -> (batch, hidden_size)
        last_output = out[:, -1, :]

        # Если задана активация, применяем её перед линейным слоем
        if self.activation is not None:
            last_output = self.activation(last_output)

        # Линейный слой даёт финальный прогноз (batch, 1)
        prediction = self.fc(last_output)

        return prediction
