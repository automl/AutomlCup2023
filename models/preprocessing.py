from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import transforms


class InputTransformer:
    def __init__(self, metadata, normalization_technique='standardize'):
        self.metadata = metadata
        input_shape = metadata.input_shape
        self.transforms = {}
        self.apply_image_transforms = input_shape.width > 1 and input_shape.height > 1
        self.apply_resize_image = self.apply_image_transforms \
                                  and (input_shape.width > 224 or input_shape.height > 224) \
                                  and input_shape.max_sequence_len == 1
        self.apply_1d_transforms = sum(
            [input_shape.max_sequence_len > 1,
             input_shape.channels > 1,
             input_shape.width > 1,
             input_shape.height > 1]) == 1
        self.transformed_metadata = None

        self.normalization_technique = normalization_technique
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        self.label_encoder = None
        self.num_classes_one_hot = None

        self.informative_columns = None

    def fit(self, x):
        transformed_metadata = deepcopy(self.metadata)
        if self.apply_resize_image:
            transformed_metadata.input_shape.width = min(self.metadata.input_shape.width, 224)
            transformed_metadata.input_shape.height = min(self.metadata.input_shape.height, 224)

        if self.apply_image_transforms:
            # if resize is applied these values might differ a little from the resized statistics
            if self.normalization_technique == 'normalize':
                self.mean = np.mean(x, axis=(0, 1, 3, 4))
                self.std = np.std(x, axis=(0, 1, 3, 4))
            elif self.normalization_technique == 'standardize':
                self.min = np.min(x)
                self.max = np.max(x)

        if self.apply_1d_transforms:
            self.informative_columns = np.var(np.squeeze(x), axis=0) != 0

            if self.metadata.input_shape.max_sequence_len > 1:
                transformed_metadata.input_shape.max_sequence_len = sum(self.informative_columns)
                x_reduced = x[:, self.informative_columns, :, :, :]
            elif self.metadata.input_shape.channels > 1:
                transformed_metadata.input_shape.channels = sum(self.informative_columns)
                x_reduced = x[:, :, self.informative_columns, :, :]
            elif self.metadata.input_shape.width > 1:
                transformed_metadata.input_shape.width = sum(self.informative_columns)
                x_reduced = x[:, :, :, self.informative_columns, :]
            elif self.metadata.input_shape.height > 1:
                transformed_metadata.input_shape.height = sum(self.informative_columns)
                x_reduced = x[:, :, :, :, self.informative_columns]

            unique_values = np.unique(x_reduced)
            # If we have time data that is all categorical (Only time dimension is != 1 !!!)
            if self.metadata.input_shape.max_sequence_len > 1 and all([value.is_integer() for value in unique_values]):
                self.label_encoder = LabelEncoder()
                # make sure only 0 gets mapped to 0 since we treat it as a masking token for sequences.
                encoder_values = np.append([0], unique_values)
                self.num_classes_one_hot = len(np.unique(encoder_values))
                transformed_metadata.input_shape.channels = self.num_classes_one_hot - 1
                self.label_encoder.fit(encoder_values)

        self.transformed_metadata = transformed_metadata
        return self.transformed_metadata

    def transform(self, x):
        print('Start transforming inputs.')
        if self.apply_image_transforms:
            x = self._image_transform(x)
        if self.apply_1d_transforms:
            x = self._1d_transform(x)
        print('Finished transforming inputs.')
        return x

    def _image_transform(self, x):
        assert self.apply_image_transforms
        if self.apply_resize_image:
            resizer = transforms.Resize((self.transformed_metadata.input_shape.width,
                                         self.transformed_metadata.input_shape.height), antialias=True)
            x = np.squeeze(x, axis=1)
            transformed_x = []
            step_size = 50
            for step in range(0, len(x), step_size):
                transformed_x.append(resizer(torch.Tensor(x[step:step + step_size])))
            x = torch.cat(transformed_x, dim=0)
            x = torch.unsqueeze(x, dim=1)
        x = torch.Tensor(x)
        # 'standardize
        if self.normalization_technique == 'normalize':
            normalizer = transforms.Normalize(self.mean, self.std)
            x = normalizer(x)
        elif self.normalization_technique == 'standardize':
            x = (x - self.min) / (self.max - self.min)
        return np.array(x)

    def _1d_transform(self, x):
        assert self.apply_1d_transforms
        input_shape = self.transformed_metadata.input_shape
        x = np.squeeze(x)

        if not all(self.informative_columns):
            x = x[:, self.informative_columns]

        if self.label_encoder is not None:
            num_datapoints = len(x)
            x = self.label_encoder.transform(x.reshape(-1)).reshape((num_datapoints, -1))
            x = torch.Tensor(x)
            x = F.one_hot(x.to(torch.int64), num_classes=self.num_classes_one_hot)
            x = x[:, :, 1:]  # remove the first entry of one-hot-encoding so that 0 gets encoded as the 0 vector
            x = np.array(x)

        x = x.reshape((len(x),
                       input_shape.max_sequence_len,
                       input_shape.channels,
                       input_shape.width,
                       input_shape.height,
                       ))
        return x
