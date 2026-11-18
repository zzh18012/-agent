from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


class ForwardModelError(ValueError):
    pass


@dataclass(frozen=True)
class ForwardModel:
    model_id: str
    version: str
    parameter_names: tuple[str, ...]
    description: str

    def predict(self, x_values: list[float], parameters: list[float]) -> list[float]:
        raise NotImplementedError

    @property
    def implementation_hash(self) -> str:
        payload = f"{self.model_id}:{self.version}:{self.parameter_names}:{self.description}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExponentialDecay(ForwardModel):
    def __init__(self) -> None:
        super().__init__(
            model_id="registered.exponential_decay",
            version="1",
            parameter_names=("amplitude", "decay", "offset"),
            description="y = amplitude * exp(-decay * x) + offset",
        )

    def predict(self, x_values: list[float], parameters: list[float]) -> list[float]:
        amplitude, decay, offset = parameters
        if decay < 0:
            raise ForwardModelError("decay must be nonnegative")
        return [amplitude * math.exp(-decay * value) + offset for value in x_values]


_MODELS: dict[str, ForwardModel] = {
    "registered.exponential_decay": ExponentialDecay(),
}


def get_forward_model(model_id: str) -> ForwardModel:
    try:
        return _MODELS[model_id]
    except KeyError as exc:
        raise ForwardModelError(f"unknown registered forward model: {model_id}") from exc


def forward_model_registry() -> dict[str, ForwardModel]:
    return dict(_MODELS)


def predict_array(model: ForwardModel, x_values, parameters) -> np.ndarray:
    return np.asarray(
        model.predict(list(map(float, x_values)), list(map(float, parameters))), dtype=float
    )


__all__ = [
    "ExponentialDecay",
    "ForwardModel",
    "ForwardModelError",
    "forward_model_registry",
    "get_forward_model",
    "predict_array",
]
