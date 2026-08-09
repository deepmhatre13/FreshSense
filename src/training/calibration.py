"""Confidence calibration utilities for FreshSense."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration."""

    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-6)

    def calibrate(self, val_loader: DataLoader, model: nn.Module, device: torch.device) -> float:
        """Fit temperature on validation data."""
        self.to(device)
        model.eval()
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                logits_list.append(logits)
                labels_list.append(y)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def eval_loss() -> float:
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss.item()

        optimizer.step(eval_loss)
        return float(self.temperature.item())
