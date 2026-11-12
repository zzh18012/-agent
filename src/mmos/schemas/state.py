from datetime import datetime

from mmos.schemas.common import MMOSModel
from mmos.schemas.states import ProjectState


class StateTransitionRecord(MMOSModel):
    """One append-only entry in manifests/state_history.jsonl."""

    project_id: str
    from_state: ProjectState
    to_state: ProjectState
    reason: str
    at: datetime
