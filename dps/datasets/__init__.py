from .load import load_emnist, emnist_classes, load_omniglot, omniglot_classes, load_backgrounds, background_names
from .base import (
    Dataset, ImageDataset, EmnistDataset, PatchesDataset, VisualArithmeticDataset,
    VisualArithmeticDatasetColour, GridArithmeticDataset, OmniglotDataset, OmniglotCountingDataset,
    GridOmniglotDataset, SalienceDataset, EMNIST_ObjectDetection, GridEMNIST_ObjectDetection
)
from .atari import AtariAutoencodeDataset