from datetime import datetime

from mmos.schemas.common import MMOSModel


class MLflowRunRecord(MMOSModel):
    """Local pointer from an experiment directory to its MLflow run."""

    run_id: str
    tracking_uri: str
    experiment_name: str
    logged_at: datetime
