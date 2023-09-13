"""Auto ML Cup dataset metadata."""

from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass(slots=True)
class InputShape:
    """Input shape."""

    num_examples: int
    max_sequence_len: int
    channels: int
    width: int
    height: int


OutputShape = List[int]


class OutputType(Enum):
    """Output type."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class EvaluationMetric(Enum):
    """Evaluation metric."""

    ACCURACY = "accuracy"
    MAE = "mae"
    BCE = "bce"
    RMSE = "rmse"
    F1 = "f1"
    MSE = "mse"


@dataclass(slots=True)
class AutoMLCupMetadata:
    """Auto ML Cup dataset metadata."""

    input_dimension: int
    input_shape: InputShape
    output_shape: OutputShape
    output_type: OutputType
    evaluation_metric: EvaluationMetric
    training_limit_sec: int
