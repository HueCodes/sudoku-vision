#!/usr/bin/env python3
"""
Data augmentation pipeline.

Expands training data with various augmentations to improve model robustness.
Reads from organized dataset and creates augmented versions.

Usage:
    python tools/augment_data.py data/labeled/train --output data/augmented/train
    python tools/augment_data.py data/labeled/train --multiplier 10
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Augmentation Functions
# ============================================================================

def rotate(
    image: NDArray[np.uint8],
    angle_range: tuple[float, float] = (-15, 15),
) -> NDArray[np.uint8]:
    """Rotate image by a random angle."""
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def perspective_warp(
    image: NDArray[np.uint8],
    distortion: float = 0.1,
) -> NDArray[np.uint8]:
    """Apply random perspective distortion."""
    h, w = image.shape[:2]

    # Original corners
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Random distortion
    dst = src.copy()
    for i in range(4):
        dst[i][0] += random.uniform(-distortion, distortion) * w
        dst[i][1] += random.uniform(-distortion, distortion) * h

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def adjust_brightness(
    image: NDArray[np.uint8],
    range_pct: tuple[float, float] = (-0.2, 0.2),
) -> NDArray[np.uint8]:
    """Adjust image brightness."""
    delta = random.uniform(*range_pct) * 255
    adjusted = np.clip(image.astype(np.float32) + delta, 0, 255)
    return adjusted.astype(np.uint8)


def adjust_contrast(
    image: NDArray[np.uint8],
    range_factor: tuple[float, float] = (0.8, 1.2),
) -> NDArray[np.uint8]:
    """Adjust image contrast."""
    factor = random.uniform(*range_factor)
    mean = np.mean(image)
    adjusted = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255)
    return adjusted.astype(np.uint8)


def gaussian_blur(
    image: NDArray[np.uint8],
    kernel_range: tuple[int, int] = (1, 3),
) -> NDArray[np.uint8]:
    """Apply Gaussian blur."""
    k = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)


def add_noise(
    image: NDArray[np.uint8],
    noise_type: str = "gaussian",
    intensity: float = 0.05,
) -> NDArray[np.uint8]:
    """Add random noise to image."""
    if noise_type == "gaussian":
        noise = np.random.normal(0, intensity * 255, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    elif noise_type == "salt_pepper":
        noisy = image.copy()
        # Salt
        salt_mask = np.random.random(image.shape) < intensity / 2
        noisy[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(image.shape) < intensity / 2
        noisy[pepper_mask] = 0
        return noisy

    return image


def elastic_transform(
    image: NDArray[np.uint8],
    alpha: float = 10,
    sigma: float = 3,
) -> NDArray[np.uint8]:
    """Apply elastic deformation."""
    h, w = image.shape[:2]

    # Random displacement fields
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32),
        (0, 0), sigma,
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32),
        (0, 0), sigma,
    ) * alpha

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def random_erasing(
    image: NDArray[np.uint8],
    probability: float = 0.5,
    area_range: tuple[float, float] = (0.02, 0.1),
) -> NDArray[np.uint8]:
    """Randomly erase a rectangular region."""
    if random.random() > probability:
        return image

    h, w = image.shape[:2]
    area = h * w

    target_area = random.uniform(*area_range) * area
    aspect_ratio = random.uniform(0.3, 3.0)

    eh = int(round(np.sqrt(target_area * aspect_ratio)))
    ew = int(round(np.sqrt(target_area / aspect_ratio)))

    if eh >= h or ew >= w:
        return image

    x = random.randint(0, w - ew)
    y = random.randint(0, h - eh)

    erased = image.copy()
    erased[y:y+eh, x:x+ew] = random.randint(0, 255)

    return erased


def translate(
    image: NDArray[np.uint8],
    max_shift: float = 0.1,
) -> NDArray[np.uint8]:
    """Translate image by random amount."""
    h, w = image.shape[:2]
    dx = int(random.uniform(-max_shift, max_shift) * w)
    dy = int(random.uniform(-max_shift, max_shift) * h)

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return translated


def scale(
    image: NDArray[np.uint8],
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> NDArray[np.uint8]:
    """Scale image by random factor."""
    factor = random.uniform(*scale_range)
    h, w = image.shape[:2]

    new_h, new_w = int(h * factor), int(w * factor)
    scaled = cv2.resize(image, (new_w, new_h))

    # Pad or crop to original size
    if factor > 1:
        # Crop
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        result = scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        # Pad
        result = np.full((h, w), 128, dtype=np.uint8)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        result[start_y:start_y+new_h, start_x:start_x+new_w] = scaled

    return result


# ============================================================================
# Augmentation Pipelines
# ============================================================================

def create_augmentation_pipeline(intensity: str = "medium") -> list[Callable]:
    """Create augmentation pipeline based on intensity level."""
    if intensity == "light":
        return [
            lambda img: rotate(img, (-5, 5)),
            lambda img: adjust_brightness(img, (-0.1, 0.1)),
            lambda img: adjust_contrast(img, (0.9, 1.1)),
        ]

    elif intensity == "medium":
        return [
            lambda img: rotate(img, (-10, 10)),
            lambda img: perspective_warp(img, 0.05),
            lambda img: adjust_brightness(img, (-0.15, 0.15)),
            lambda img: adjust_contrast(img, (0.85, 1.15)),
            lambda img: gaussian_blur(img, (1, 3)),
            lambda img: translate(img, 0.05),
        ]

    elif intensity == "heavy":
        return [
            lambda img: rotate(img, (-15, 15)),
            lambda img: perspective_warp(img, 0.1),
            lambda img: adjust_brightness(img, (-0.2, 0.2)),
            lambda img: adjust_contrast(img, (0.8, 1.2)),
            lambda img: gaussian_blur(img, (1, 5)),
            lambda img: add_noise(img, "gaussian", 0.03),
            lambda img: elastic_transform(img, 8, 3),
            lambda img: translate(img, 0.1),
            lambda img: scale(img, (0.9, 1.1)),
            lambda img: random_erasing(img, 0.3),
        ]

    else:
        raise ValueError(f"Unknown intensity: {intensity}")


def augment_image(
    image: NDArray[np.uint8],
    augmentations: list[Callable],
    n_augmentations: int = 3,
) -> NDArray[np.uint8]:
    """Apply random subset of augmentations to image."""
    result = image.copy()

    # Select random augmentations
    selected = random.sample(
        augmentations,
        min(n_augmentations, len(augmentations)),
    )

    for aug in selected:
        result = aug(result)

    return result


def process_directory(
    input_dir: Path,
    output_dir: Path,
    multiplier: int = 10,
    intensity: str = "medium",
    preserve_originals: bool = True,
) -> dict:
    """Process all images in a class directory structure.

    Args:
        input_dir: Input directory with class subdirectories
        output_dir: Output directory for augmented images
        multiplier: Number of augmented copies per original
        intensity: Augmentation intensity (light/medium/heavy)
        preserve_originals: Include original images in output

    Returns:
        dict with processing statistics
    """
    augmentations = create_augmentation_pipeline(intensity)

    stats = {
        "originals": 0,
        "augmented": 0,
        "by_class": {},
    }

    # Find all class directories
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()]

    for class_dir in sorted(class_dirs, key=lambda d: int(d.name)):
        class_label = class_dir.name
        out_class_dir = output_dir / class_label
        out_class_dir.mkdir(parents=True, exist_ok=True)

        class_stats = {"originals": 0, "augmented": 0}

        # Process each image
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))

        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            class_stats["originals"] += 1

            # Save original
            if preserve_originals:
                dst_path = out_class_dir / f"orig_{img_path.name}"
                cv2.imwrite(str(dst_path), img)

            # Generate augmented versions
            for i in range(multiplier):
                augmented = augment_image(img, augmentations)
                dst_path = out_class_dir / f"aug_{i}_{img_path.name}"
                cv2.imwrite(str(dst_path), augmented)
                class_stats["augmented"] += 1

        stats["by_class"][class_label] = class_stats
        stats["originals"] += class_stats["originals"]
        stats["augmented"] += class_stats["augmented"]

        print(f"  Class {class_label}: {class_stats['originals']} -> {class_stats['augmented']} augmented")

    return stats


def preview_augmentations(
    image_path: Path,
    output_path: Path,
    intensity: str = "medium",
) -> None:
    """Generate preview grid showing augmentation effects."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    augmentations = create_augmentation_pipeline(intensity)

    # Create 4x4 grid
    grid_size = 4
    cell_size = 56  # 2x the 28x28 cell size for visibility
    grid = np.zeros((cell_size * grid_size, cell_size * grid_size), dtype=np.uint8)

    # Original in top-left
    resized = cv2.resize(img, (cell_size, cell_size))
    grid[0:cell_size, 0:cell_size] = resized

    # Augmented versions in remaining cells
    for i in range(1, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size

        augmented = augment_image(img, augmentations)
        resized = cv2.resize(augmented, (cell_size, cell_size))

        y = row * cell_size
        x = col * cell_size
        grid[y:y+cell_size, x:x+cell_size] = resized

    cv2.imwrite(str(output_path), grid)
    print(f"Preview saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment training data for improved model robustness"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with class subdirectories",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: input_dir_augmented)",
    )
    parser.add_argument(
        "--multiplier", "-m",
        type=int,
        default=10,
        help="Number of augmented copies per original (default: 10)",
    )
    parser.add_argument(
        "--intensity", "-i",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Augmentation intensity (default: medium)",
    )
    parser.add_argument(
        "--no-originals",
        action="store_true",
        help="Don't include original images in output",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Generate preview for a single image instead of processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Preview mode
    if args.preview:
        output_path = args.output or Path("augmentation_preview.png")
        preview_augmentations(args.preview, output_path, args.intensity)
        return

    # Full processing
    output_dir = args.output or Path(str(args.input_dir) + "_augmented")

    print("Data Augmentation Pipeline")
    print("=" * 50)
    print(f"Input: {args.input_dir}")
    print(f"Output: {output_dir}")
    print(f"Multiplier: {args.multiplier}x")
    print(f"Intensity: {args.intensity}")
    print(f"Include originals: {not args.no_originals}")
    print("=" * 50)
    print("\nProcessing...")

    stats = process_directory(
        args.input_dir,
        output_dir,
        multiplier=args.multiplier,
        intensity=args.intensity,
        preserve_originals=not args.no_originals,
    )

    print("\n" + "=" * 50)
    print("AUGMENTATION COMPLETE")
    print("=" * 50)
    print(f"Original images: {stats['originals']}")
    print(f"Augmented images: {stats['augmented']}")
    print(f"Total output: {stats['originals'] + stats['augmented'] if not args.no_originals else stats['augmented']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
