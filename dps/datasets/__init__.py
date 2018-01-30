from .load import load_emnist, emnist_classes, load_omniglot, omniglot_classes
from .base import (
    SupervisedDataset, EmnistDataset, PatchesDataset, VisualArithmeticDataset, VisualArithmeticDatasetColour,
    GridArithmeticDataset, OmniglotDataset, OmniglotCountingDataset, GridOmniglotDataset, SalienceDataset
)
from .atari import AtariAutoencodeDataset