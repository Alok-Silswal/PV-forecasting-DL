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
from models.proposed_model import ProposedModel
from models.comparison_models.cnn import CNN
from models.comparison_models.lstm import LSTM
from models.comparison_models.cnn_lstm import CNNLSTM
from models.dcnn_rbilstm import DCNNResidualBiLSTM
from models.phn_model import PHNModel

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
        return ProposedModel(**kwargs)

    if model_name == "proposed_no_ta":
        return ProposedModel(use_temporal_attention=False, **kwargs)

    if model_name == "proposed_no_fa":
        return ProposedModel(use_feature_attention=False, **kwargs)

    if model_name == "proposed_no_fusion":
        return ProposedModel(use_scalar_gated_fusion=False, **kwargs)

    if model_name == "proposed_no_fa_no_ta_no_fusion":
        return ProposedModel(
            use_feature_attention=False,
            use_temporal_attention=False,
            use_scalar_gated_fusion=False,
            **kwargs,
        )

    if model_name in {
        "proposed_phn",
        "proposed_phn_5q",
        "proposed_phn_6q",
        "proposed_phn_7q",
        "proposed_phn_8q",
        "proposed_phn_9q",
        "proposed_phn_10q",
    }:
        return ProposedModel(use_quantum_branch=True, **kwargs)

    if model_name == "phn":
        return PHNModel(**kwargs)

    if model_name == "cnn":
        return CNN(**kwargs)

    if model_name == "lstm":
        return LSTM(**kwargs)

    if model_name == "cnn_lstm":
        return CNNLSTM(**kwargs)

    if model_name == "dcnn_rbilstm":
        return DCNNResidualBiLSTM(**kwargs)

    available_models = [
        "proposed",
        "proposed_no_ta",
        "proposed_no_fa",
        "proposed_no_fusion",
        "proposed_no_fa_no_ta_no_fusion",
        "proposed_phn",
        "proposed_phn_5q",
        "proposed_phn_6q",
        "proposed_phn_7q",
        "proposed_phn_8q",
        "proposed_phn_9q",
        "proposed_phn_10q",
        "phn",
        "cnn",
        "lstm",
        "cnn_lstm",
        "dcnn_rbilstm",
    ]

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Available models: {', '.join(available_models)}."
    )