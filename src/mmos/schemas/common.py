from pydantic import BaseModel, ConfigDict


class MMOSModel(BaseModel):
    """Base model for all persisted or cross-module data.

    Unknown fields are rejected so that a typo or a schema drift fails loudly
    instead of being silently dropped.
    """

    model_config = ConfigDict(extra="forbid")
