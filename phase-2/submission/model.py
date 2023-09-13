"""User model."""

from typing import Dict
import numpy as np
import torch
from numpy.typing import ArrayLike

from metadata import *
import xgboost as xgb 


class Model:
    """User model."""

    def __init__(self, metadata):
        """Initialize the model."""
        self.metadata = metadata
        print(f"FROM MODEL.PY: METADATA {metadata}")
        print(f"INPUT_DIMENSION {metadata.input_dimension}")
        print(f"NUM_EXAMPLES {metadata.input_shape.num_examples}")
        print(f"MAX_SEQUENCE_LEN {metadata.input_shape.max_sequence_len}")
        print(f"CHANNELS {metadata.input_shape.channels}")
        print(f"WIDTH {metadata.input_shape.width}")
        print(f"HEIGHT {metadata.input_shape.height}")
        print(f"OUTPUT_SHAPE {metadata.output_shape}")
        print(f"OUTPUT_TYPE {metadata.output_type}")
        print(f"EVALUATION_METRIC {metadata.evaluation_metric}")
        if metadata.output_type == OutputType.CLASSIFICATION:
            self.sub_model = xgb.XGBClassifier()
        else:
            self.sub_model = xgb.XGBRegressor()

    def train(self, train_dataset: Dict[str, ArrayLike]):
        """Trains the model.

        Args:
            train_dataset (Dict[str, ArrayLike]): The training dataset.
        """
        print("FROM MODEL.PY: TRAIN")
        print(f"FROM MODEL.PY: CUDA {torch.cuda.is_available()}")
        print(f"FROM MODEL.PY: SELF {self}")

        try:
            train_input = np.array(train_dataset["input"])
        except ValueError:
            max_len = max(map(len, train_dataset["input"]))
            train_input = np.zeros((len(train_dataset), max_len), dtype=np.float32)
            for idx, row in enumerate(train_dataset["input"]):
                train_input[idx, : len(row)] = row

        train_label = np.array(train_dataset["label"])

        train_input = train_input.reshape((train_input.shape[0],-1))

        self.sub_model.fit(train_input, train_label)

        print(f"FROM MODEL.PY: INPUT SHAPE {train_input.shape}")
        print(f"FROM MODEL.PY: LABEL SHAPE {train_label.shape}")

    def predict(self, prediction_dataset: Dict[str, ArrayLike]) -> ArrayLike:
        """Predicts over a prediction dataset using the model.

        Args:
            prediction_dataset (Dict[str, ArrayLike]): Dataset to use for prediction.

        Returns:
            ArrayLike: The predictions.
        """
        print("FROM MODEL.PY: PREDICT")
        print(f"FROM MODEL.PY: SELF {self}")

        #print(((len(prediction_dataset["input"]), *self.metadata.output_shape)))

        test_input = prediction_dataset["input"]
        test_input = test_input.reshape((test_input.shape[0],-1))
        return self.sub_model.predict(test_input)#np.zeros((len(prediction_dataset["input"]), *self.metadata.output_shape))
