"""
model_factory.py

Centralized factory for constructing forecasting models.

This module provides a single entry point for instantiating forecasting
architectures used throughout the project. Keeping model creation here
allows training and evaluation pipelines to remain model-agnostic while
making it straightforward to add new architectures in the future.
"""

import json

from torch import nn

from configs import config
from models.proposed_model import ProposedModel
from models.comparison_models.cnn import CNN
from models.comparison_models.lstm import LSTM
from models.comparison_models.cnn_lstm import CNNLSTM

def _load_proposed_hpo_kwargs() -> dict:
    """
    Load the finalized HPO-selected constructor arguments for
    ``proposed_hpo``.

    Reads ``best_hpo.json`` as produced by ``hpo.finalize_hpo`` for the
    proposed model at the active forecast horizon. This is always read
    from the ``proposed`` model's HPO directory (never from
    ``proposed_hpo``'s own experiment directory), since HPO is only
    ever run for the proposed model; ``proposed_hpo`` is a distinct
    *retraining* target that reuses those results at
    ``config.ACTIVE_HORIZON``.

    Returns
    -------
    dict
        The ``"best_hyperparameters"`` mapping from ``best_hpo.json``,
        forwarded directly as ``ProposedModel`` constructor arguments.

    Raises
    ------
    FileNotFoundError
        If ``best_hpo.json`` does not exist for the active horizon.
        Run ``python run_hpo.py --optimizer ...`` for each optimizer,
        then ``python run_hpo.py --finalize``, before selecting
        ``proposed_hpo``.
    KeyError
        If ``best_hpo.json`` does not contain a
        ``"best_hyperparameters"`` key.
    """

    horizon_dir_name = f"horizon_{config.ACTIVE_HORIZON}"
    best_hpo_path = (
        config.EXPERIMENTS_DIR / "proposed" / horizon_dir_name / "hpo" / "best_hpo.json"
    )

    if not best_hpo_path.exists():
        raise FileNotFoundError(
            f"best_hpo.json not found at {best_hpo_path}. Run HPO for "
            f"every optimizer and then 'python run_hpo.py --finalize' "
            f"before selecting 'proposed_hpo'."
        )

    with open(best_hpo_path, "r", encoding="utf-8") as best_hpo_file:
        best_hpo = json.load(best_hpo_file)

    if "best_hyperparameters" not in best_hpo:
        raise KeyError(
            f"{best_hpo_path} is missing the required "
            f"'best_hyperparameters' key."
        )

    return best_hpo["best_hyperparameters"]


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

    if model_name == "proposed_no_fa":
        return ProposedModel(use_feature_attention=False, **kwargs)

    if model_name == "proposed_no_ta":
        return ProposedModel(use_temporal_attention=False, **kwargs)

    if model_name == "proposed_no_fusion":
        return ProposedModel(use_scalar_gated_fusion=False, **kwargs)

    if model_name == "proposed_hpo":
        hpo_kwargs = _load_proposed_hpo_kwargs()
        hpo_kwargs.update(kwargs)
        return ProposedModel(**hpo_kwargs)

    if model_name == "cnn":
        return CNN(**kwargs)

    if model_name == "lstm":
        return LSTM(**kwargs)

    if model_name == "cnn_lstm":
        return CNNLSTM(**kwargs)

    available_models = [
        "proposed",
        "proposed_no_fa",
        "proposed_no_ta",
        "proposed_no_fusion",
        "proposed_hpo",
        "cnn",
        "lstm",
        "cnn_lstm",
    ]

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Available models: {', '.join(available_models)}."
    )