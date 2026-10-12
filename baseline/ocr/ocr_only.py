"""Ground-truth crops keep detection errors out of OCR-only evaluation."""

from .io import dataset_path


def input_path(dataset, sample):
    return dataset_path(dataset, sample["crop_path"])


def prepare_image(image, sample):
    return image, {}
