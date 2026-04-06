"""Portable, single-file DART hand-pose data loader.

Usage:
    - In this repo: ``from production_lib.dart_data_loader import PortableDartDataset``
    - If copied to another repo: ``from dart_data_loader import PortableDartDataset``

Supported dataset layouts:

1) Flat labels format (primary; used by this repo training scripts):

"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

HAND_KEYPOINT_NAMES_21 = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# Common DART 26-point -> 21-point hand-only mapping.
DART_26_TO_HAND_21 = (
    0,
    22,
    23,
    24,
    25,
    17,
    18,
    19,
    20,
    12,
    13,
    14,
    15,
    7,
    8,
    9,
    10,
    2,
    3,
    4,
    5,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _first_existing_key(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def to_hand_21(keypoints: Any) -> np.ndarray:
    """Normalize keypoints to the 21-point hand layout.

    Accepts arrays shaped (21, C) or (26, C).
    """

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()

    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected keypoints with shape (N, C), got {arr.shape}")

    if arr.shape[0] == 21:
        return arr.copy()
    if arr.shape[0] == 26:
        return arr[np.array(DART_26_TO_HAND_21)]

    raise ValueError(f"Unsupported keypoint count {arr.shape[0]}; expected 21 or 26")


def crop_from_keypoints(
    image: Image.Image,
    keypoints_xy: np.ndarray,
    padding: int = 20,
) -> tuple[Image.Image, np.ndarray]:
    """Crop image around keypoints and return adjusted keypoints in cropped coordinates."""

    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    if keypoints_xy.ndim != 2 or keypoints_xy.shape[1] < 2:
        raise ValueError("keypoints_xy must have shape (N, >=2)")

    width, height = image.size

    min_x = max(int(np.floor(np.min(keypoints_xy[:, 0]))) - padding, 0)
    min_y = max(int(np.floor(np.min(keypoints_xy[:, 1]))) - padding, 0)
    max_x = min(int(np.ceil(np.max(keypoints_xy[:, 0]))) + padding, width)
    max_y = min(int(np.ceil(np.max(keypoints_xy[:, 1]))) + padding, height)

    if max_x <= min_x or max_y <= min_y:
        return image.copy(), keypoints_xy.copy()

    cropped = image.crop((min_x, min_y, max_x, max_y))
    adjusted = keypoints_xy.copy()
    adjusted[:, 0] -= min_x
    adjusted[:, 1] -= min_y
    return cropped, adjusted


class PortableDartDataset(Dataset):
    """Reusable dataset for DART-style folders.

    Returns a dict per sample with keys:
      - image: PIL.Image or transformed output
      - keypoints_2d: torch.FloatTensor [21,2] or None
      - keypoints_3d: torch.FloatTensor [21,3] or None
      - image_path: str
      - sequence_id: str
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        transform=None,
        crop_size: int | None = 180,
        bbox_padding: int = 20,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.crop_size = crop_size
        self.bbox_padding = bbox_padding
        self.samples: list[dict[str, Any]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_dir}"
            )

        self._load_flat_labels_dataset()
        if not self.samples:
            self._load_sequence_folder_dataset()

        if not self.samples:
            raise RuntimeError(
                "No samples found. Expected either:\n"
                "  (1) root/images + root/labels.pkl, or\n"
                "  (2) sequence subfolders with output.pkl + images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        keypoints_2d = sample["keypoints_2d"]
        if keypoints_2d is not None:
            image, keypoints_2d = crop_from_keypoints(
                image=image,
                keypoints_xy=keypoints_2d,
                padding=self.bbox_padding,
            )

        if self.crop_size is not None:
            src_w, src_h = image.size
            image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            if keypoints_2d is not None and src_w > 0 and src_h > 0:
                sx = self.crop_size / float(src_w)
                sy = self.crop_size / float(src_h)
                keypoints_2d = keypoints_2d.copy()
                keypoints_2d[:, 0] *= sx
                keypoints_2d[:, 1] *= sy

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "keypoints_2d": torch.tensor(keypoints_2d, dtype=torch.float32)
            if keypoints_2d is not None
            else None,
            "keypoints_3d": torch.tensor(sample["keypoints_3d"], dtype=torch.float32)
            if sample["keypoints_3d"] is not None
            else None,
            "image_path": sample["image_path"],
            "sequence_id": sample["sequence_id"],
        }

    def _load_flat_labels_dataset(self) -> None:
        """Load datasets organized as root/images + root/labels.pkl."""
        label_path = self.root_dir / "labels.pkl"
        image_dir = self.root_dir / "images"
        if not (label_path.exists() and image_dir.is_dir()):
            return

        with label_path.open("rb") as handle:
            labels = pickle.load(handle)

        image_names = _first_existing_key(labels, ("img", "images", "image_names"))
        all_2d = _first_existing_key(labels, ("joint2d", "joint_2d"))
        all_3d = _first_existing_key(labels, ("joint3d", "joint_3d"))
        if image_names is None or all_2d is None:
            return

        for img_name in image_names:
            image_path = image_dir / img_name
            if not image_path.exists():
                continue

            label_index = int(str(img_name).split("_")[0])
            keypoints_2d = to_hand_21(all_2d[label_index])[:, :2]
            keypoints_3d = None
            if all_3d is not None:
                keypoints_3d = to_hand_21(all_3d[label_index])[:, :3]

            self.samples.append(
                {
                    "image_path": str(image_path),
                    "sequence_id": "flat_labels",
                    "keypoints_2d": keypoints_2d,
                    "keypoints_3d": keypoints_3d,
                }
            )

    def _load_sequence_folder_dataset(self) -> None:
        """Load datasets organized as subfolders each with output.pkl + images."""
        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue

            label_path = folder / "output.pkl"
            if not label_path.exists():
                continue

            with label_path.open("rb") as handle:
                labels = pickle.load(handle)

            raw_2d = _first_existing_key(labels, ("joint_2d", "joint2d"))
            raw_3d = _first_existing_key(labels, ("joint_3d", "joint3d"))
            keypoints_2d = to_hand_21(raw_2d)[:, :2] if raw_2d is not None else None
            keypoints_3d = to_hand_21(raw_3d)[:, :3] if raw_3d is not None else None

            for image_path in sorted(folder.iterdir()):
                if (
                    image_path.suffix.lower() not in IMAGE_EXTENSIONS
                    or not image_path.is_file()
                ):
                    continue
                self.samples.append(
                    {
                        "image_path": str(image_path),
                        "sequence_id": folder.name,
                        "keypoints_2d": keypoints_2d.copy()
                        if keypoints_2d is not None
                        else None,
                        "keypoints_3d": keypoints_3d.copy()
                        if keypoints_3d is not None
                        else None,
                    }
                )


__all__ = [
    "PortableDartDataset",
    "to_hand_21",
    "crop_from_keypoints",
    "HAND_KEYPOINT_NAMES_21",
]
