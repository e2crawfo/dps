from .load import load_emnist, emnist_classes, load_omniglot, omniglot_classes
from .base import (
    EmnistDataset, PatchesDataset, VisualArithmeticDataset, GridArithmeticDataset,
    OmniglotDataset, OmniglotCountingDataset, GridOmniglotDataset, SalienceDataset
)
from .atari import AtariAutoEncodeDataset