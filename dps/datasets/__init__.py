from .load import load_emnist, emnist_classes, load_omniglot, omniglot_classes
from .base import (
    SupervisedDataset, AutoencodeDataset, EmnistDataset, PatchesDataset, VisualArithmeticDataset,
    VisualArithmeticDatasetColour, GridArithmeticDataset, OmniglotDataset, OmniglotCountingDataset,
    GridOmniglotDataset, SalienceDataset, EMNIST_ObjectDetection
)
from .atari import AtariAutoencodeDataset