"""Evaluate Proposed-Reupload without training or refitting scalers."""

from models.proposed_rvqc.evaluate import main
from . import ProposedReupload


if __name__ == "__main__":
    main(ProposedReupload, "proposed_reupload")
