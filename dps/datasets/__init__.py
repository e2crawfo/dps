from .load import load_emnist, emnist_classes, load_omniglot, omniglot_classes, load_backgrounds, background_names
from .base import (
    Dataset, tf_image_representation, ImageClassification, Emnist, Omniglot,
    Patches, GridPatches, VisualArithmetic, GridArithmetic,
    tf_annotation_representation, EmnistObjectDetection, GridEmnistObjectDetection,
)
from .atari import AtariAutoencode