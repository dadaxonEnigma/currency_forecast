# src/model/model_lstm.py
"""
LSTM-модель для прогноза курса USD→UZS.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # берем последнее скрытое состояние
        last_output = out[:, -1, :]
        out = self.fc(last_output)
        return out
