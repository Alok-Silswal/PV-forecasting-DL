"""
model_factory.py

Centralized factory for constructing forecasting models.

This module provides a single entry point for instantiating forecasting
architectures used throughout the project. Keeping model creation here
allows training and evaluation pipelines to remain model-agnostic while
making it straightforward to add new architectures in the future.
"""

from torch import nn

from configs import config
from models.proposed_no_quantum import ProposedNoQuantum
from models.proposed_model import ResidualForecast
from models.comparison_models.cnn import CNN
from models.comparison_models.lstm import LSTM
from models.comparison_models.cnn_lstm import CNNLSTM
from models.comparison_models.tcn import TCN
from models.dcnn_rbilstm import DCNNResidualBiLSTM

MODEL_NAMES = (
    "proposed", "proposed_no_quantum", "tcn",
    "cnn", "lstm", "cnn_lstm", "dcnn_rbilstm",
    "proposed_no_ta", "proposed_no_fa", "proposed_no_fusion",
    "proposed_no_fa_no_ta_no_fusion",
)

def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Construct and return the requested forecasting model.

    Parameters
    ----------
    model_name : str
        Name of the forecasting model to instantiate. Model names are
        case-insensitive.

    **kwargs
        Optional keyword arguments forwarded to the model constructor.

    Returns
    -------
    nn.Module
        Instantiated forecasting model.

    Raises
    ------
    ValueError
        If the requested model is not supported.
    """

    model_name = model_name.lower()

    if model_name == "proposed":
        return ResidualForecast(**kwargs)

    if model_name == "proposed_no_quantum":
        return ProposedNoQuantum(**kwargs)

    if model_name == "tcn":
        return TCN(**kwargs)

    if model_name == "proposed_no_ta":
        return ProposedNoQuantum(use_temporal_attention=False, **kwargs)

    if model_name == "proposed_no_fa":
        return ProposedNoQuantum(use_feature_attention=False, **kwargs)

    if model_name == "proposed_no_fusion":
        return ProposedNoQuantum(use_scalar_gated_fusion=False, **kwargs)

    if model_name == "proposed_no_fa_no_ta_no_fusion":
        return ProposedNoQuantum(
            use_feature_attention=False,
            use_temporal_attention=False,
            use_scalar_gated_fusion=False,
            **kwargs,
        )

    if model_name == "cnn":
        return CNN(**kwargs)

    if model_name == "lstm":
        return LSTM(**kwargs)

    if model_name == "cnn_lstm":
        return CNNLSTM(**kwargs)

    if model_name == "dcnn_rbilstm":
        return DCNNResidualBiLSTM(**kwargs)

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Available models: {', '.join(MODEL_NAMES)}."
    )
