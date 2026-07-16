"""
Loss function for short-term photovoltaic (PV) power forecasting.

This module provides the loss function used during model training and
validation. The training objective is Mean Squared Error (MSE), which is the standard regression loss for PV forecasting.

"""

import torch.nn as nn


def get_loss_function() -> nn.Module:

    return nn.MSELoss()