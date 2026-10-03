"""Train Proposed-Reupload using the matched RVQC training protocol."""

from models.proposed_rvqc.run_experiment import main
from . import ProposedReupload


if __name__ == "__main__":
    main(ProposedReupload, "proposed_reupload")
