from .load import load_emnist, emnist_classes, load_omniglot, omniglot_classes, load_backgrounds, background_names, hard_background_names
from .base import (
    ArrayFeature, ImageFeature, Dataset, ImageDataset,
    ImageClassificationDataset, EmnistDataset, OmniglotDataset, PatchesDataset, GridPatchesDataset,
    VisualArithmeticDataset, GridArithmeticDataset,
    EmnistObjectDetectionDataset, GridEmnistObjectDetectionDataset,
)
from .game import GameDataset