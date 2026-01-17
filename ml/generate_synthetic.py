#!/usr/bin/env python3
"""
Synthetic Data Generator for Sudoku Digit Recognition

Generates training images of printed digits (1-9) and empty cells (0).
Uses various fonts, sizes, and augmentations to simulate real sudoku puzzles.

Usage:
    python generate_synthetic.py --output ../data/synthetic --num-per-class 5000
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Fonts to use for digit generation (common fonts available on macOS)
FONTS = [
    # Sans-serif (most common in sudoku)
    "Arial",
    "Arial Black",
    "Arial Narrow",
    "Helvetica",
    "Helvetica Neue",
    "Verdana",
    "Tahoma",
    # Serif
    "Times New Roman",
    "Georgia",
    # Monospace
    "Courier New",
    "Menlo",
    "Monaco",
]

# Font styles to try
FONT_STYLES = ["Regular", "Bold", ""]


def find_font(font_name: str, size: int) -> ImageFont.FreeTypeFont | None:
    """Try to load a font by name."""
    for style in FONT_STYLES:
        try:
            if style:
                full_name = f"{font_name} {style}"
            else:
                full_name = font_name
            return ImageFont.truetype(full_name, size)
        except (OSError, IOError):
            continue

    # Try common paths
    font_paths = [
        f"/System/Library/Fonts/{font_name}.ttf",
        f"/System/Library/Fonts/{font_name}.ttc",
        f"/Library/Fonts/{font_name}.ttf",
        f"/Library/Fonts/{font_name}.ttc",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    return None


def get_available_fonts(size: int = 24) -> list[ImageFont.FreeTypeFont]:
    """Get list of available fonts."""
    fonts = []
    for font_name in FONTS:
        font = find_font(font_name, size)
        if font:
            fonts.append((font_name, font))
    return fonts


def generate_digit_image(
    digit: int,
    font: ImageFont.FreeTypeFont,
    size: int = 28,
    jitter: int = 2,
) -> np.ndarray:
    """Generate a single digit image.

    Args:
        digit: Digit to render (1-9)
        font: PIL font to use
        size: Output image size
        jitter: Random position offset in pixels

    Returns:
        Grayscale numpy array (size x size)
    """
    # Create image with padding for rendering
    padding = 10
    img_size = size + padding * 2
    img = Image.new("L", (img_size, img_size), color=255)  # White background
    draw = ImageDraw.Draw(img)

    # Get text bounding box
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center with jitter
    x = (img_size - text_width) // 2 + random.randint(-jitter, jitter)
    y = (img_size - text_height) // 2 + random.randint(-jitter, jitter) - bbox[1]

    # Draw digit in black
    draw.text((x, y), text, font=font, fill=0)

    # Crop to target size (center crop)
    left = padding
    top = padding
    img = img.crop((left, top, left + size, top + size))

    return np.array(img)


def generate_empty_cell(size: int = 28) -> np.ndarray:
    """Generate an empty cell image.

    Includes variations like:
    - Plain white/gray
    - Subtle noise
    - Faint grid artifacts
    - Slight gradients
    """
    variation = random.choice(["plain", "noisy", "gradient", "artifact"])

    if variation == "plain":
        # Plain white or light gray
        value = random.randint(240, 255)
        img = np.full((size, size), value, dtype=np.uint8)

    elif variation == "noisy":
        # White with Gaussian noise
        base = random.randint(245, 255)
        noise = np.random.normal(0, random.uniform(2, 8), (size, size))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)

    elif variation == "gradient":
        # Subtle gradient (simulates uneven lighting)
        base = random.randint(240, 255)
        img = np.full((size, size), base, dtype=np.uint8)

        # Add gradient
        direction = random.choice(["h", "v", "d"])
        gradient_strength = random.uniform(5, 15)

        if direction == "h":
            gradient = np.linspace(-gradient_strength, gradient_strength, size)
            img = np.clip(img + gradient[np.newaxis, :], 0, 255).astype(np.uint8)
        elif direction == "v":
            gradient = np.linspace(-gradient_strength, gradient_strength, size)
            img = np.clip(img + gradient[:, np.newaxis], 0, 255).astype(np.uint8)
        else:
            # Diagonal
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xx, yy = np.meshgrid(x, y)
            gradient = (xx + yy) * gradient_strength / 2
            img = np.clip(img + gradient, 0, 255).astype(np.uint8)

    else:  # artifact
        # Faint grid line artifacts at edges
        base = random.randint(245, 255)
        img = np.full((size, size), base, dtype=np.uint8)

        # Add faint lines at random edges
        line_darkness = random.randint(200, 230)
        line_width = random.randint(1, 2)

        if random.random() > 0.5:
            img[:line_width, :] = line_darkness  # Top
        if random.random() > 0.5:
            img[-line_width:, :] = line_darkness  # Bottom
        if random.random() > 0.5:
            img[:, :line_width] = line_darkness  # Left
        if random.random() > 0.5:
            img[:, -line_width:] = line_darkness  # Right

    return img


def apply_augmentation(img: np.ndarray, is_empty: bool = False) -> np.ndarray:
    """Apply random augmentations to an image.

    Args:
        img: Input grayscale image
        is_empty: Whether this is an empty cell (less aggressive augmentation)
    """
    h, w = img.shape

    # Rotation (±10 degrees)
    if random.random() > 0.3:
        angle = random.uniform(-10, 10)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=255)

    # Scale (0.85 - 1.15)
    if random.random() > 0.3:
        scale = random.uniform(0.85, 1.15)
        new_size = int(w * scale)
        if new_size > 0:
            scaled = cv2.resize(img, (new_size, new_size))
            # Center crop or pad to original size
            if new_size > w:
                start = (new_size - w) // 2
                img = scaled[start:start+h, start:start+w]
            else:
                pad = (w - new_size) // 2
                img = cv2.copyMakeBorder(scaled, pad, w-new_size-pad,
                                         pad, w-new_size-pad,
                                         cv2.BORDER_CONSTANT, value=255)
                img = cv2.resize(img, (w, h))

    # Gaussian blur
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # Brightness adjustment
    if random.random() > 0.3:
        brightness = random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

    # Contrast adjustment
    if random.random() > 0.3:
        contrast = random.uniform(0.8, 1.2)
        mean = np.mean(img)
        img = np.clip((img.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # Gaussian noise
    if random.random() > 0.5:
        noise_sigma = random.uniform(3, 12)
        noise = np.random.normal(0, noise_sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Erosion/Dilation (only for digits)
    if not is_empty and random.random() > 0.7:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # Perspective transform (subtle)
    if random.random() > 0.7:
        strength = random.uniform(1, 3)
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([
            [random.uniform(0, strength), random.uniform(0, strength)],
            [w - random.uniform(0, strength), random.uniform(0, strength)],
            [w - random.uniform(0, strength), h - random.uniform(0, strength)],
            [random.uniform(0, strength), h - random.uniform(0, strength)]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matrix, (w, h), borderValue=255)

    return img


def generate_dataset(
    output_dir: Path,
    num_per_class: int = 5000,
    val_ratio: float = 0.1,
    size: int = 28,
    seed: int = 42,
) -> dict:
    """Generate the full synthetic dataset.

    Args:
        output_dir: Directory to save images
        num_per_class: Number of images per class
        val_ratio: Fraction to use for validation
        size: Image size
        seed: Random seed

    Returns:
        Metadata dictionary
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    for split_dir in [train_dir, val_dir]:
        for digit in range(10):
            (split_dir / str(digit)).mkdir(parents=True, exist_ok=True)

    # Get available fonts
    fonts = get_available_fonts(size=random.randint(20, 28))
    print(f"Found {len(fonts)} fonts: {[f[0] for f in fonts]}")

    if not fonts:
        print("Warning: No fonts found, using default")
        fonts = [("default", ImageFont.load_default())]

    # Track statistics
    stats = {
        "total_generated": 0,
        "train_count": 0,
        "val_count": 0,
        "per_class": {str(d): {"train": 0, "val": 0} for d in range(10)},
        "fonts_used": [f[0] for f in fonts],
    }

    num_val = int(num_per_class * val_ratio)
    num_train = num_per_class - num_val

    for digit in range(10):
        print(f"Generating class {digit}...", end=" ", flush=True)

        for i in range(num_per_class):
            # Decide train or val
            is_val = i < num_val
            split_dir = val_dir if is_val else train_dir
            split_name = "val" if is_val else "train"

            # Generate image
            if digit == 0:
                # Empty cell
                img = generate_empty_cell(size=size)
                img = apply_augmentation(img, is_empty=True)
            else:
                # Digit
                font_name, font = random.choice(fonts)
                # Recreate font with random size
                font_size = random.randint(18, 26)
                font = find_font(font_name, font_size)
                if font is None:
                    font = fonts[0][1]

                img = generate_digit_image(digit, font, size=size, jitter=2)
                img = apply_augmentation(img, is_empty=False)

            # Ensure correct size
            if img.shape != (size, size):
                img = cv2.resize(img, (size, size))

            # Save
            filename = f"{digit}_{i:05d}.png"
            cv2.imwrite(str(split_dir / str(digit) / filename), img)

            # Update stats
            stats["total_generated"] += 1
            stats[f"{split_name}_count"] += 1
            stats["per_class"][str(digit)][split_name] += 1

        print(f"done ({num_train} train, {num_val} val)")

    # Save metadata
    metadata = {
        "size": size,
        "num_per_class": num_per_class,
        "val_ratio": val_ratio,
        "seed": seed,
        "stats": stats,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def generate_samples(output_dir: Path, num_samples: int = 10) -> None:
    """Generate sample images for visual verification."""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    fonts = get_available_fonts(size=24)
    if not fonts:
        fonts = [("default", ImageFont.load_default())]

    # Generate samples for each class
    for digit in range(10):
        for i in range(num_samples):
            if digit == 0:
                img = generate_empty_cell()
            else:
                font_name, font = fonts[i % len(fonts)]
                font = find_font(font_name, random.randint(18, 26)) or font
                img = generate_digit_image(digit, font)

            # Save without augmentation for clarity
            cv2.imwrite(str(samples_dir / f"sample_{digit}_{i}.png"), img)

            # Save with augmentation
            img_aug = apply_augmentation(img, is_empty=(digit == 0))
            cv2.imwrite(str(samples_dir / f"sample_{digit}_{i}_aug.png"), img_aug)

    print(f"Saved {10 * num_samples * 2} sample images to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic digit dataset")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--num-per-class", "-n",
        type=int,
        default=5000,
        help="Number of images per class",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=28,
        help="Image size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Only generate sample images for verification",
    )
    args = parser.parse_args()

    print(f"Output directory: {args.output}")

    if args.samples_only:
        generate_samples(args.output)
        return

    print(f"Generating {args.num_per_class} images per class (10 classes)")
    print(f"Total images: {args.num_per_class * 10}")
    print(f"Train/val split: {1 - args.val_ratio:.0%}/{args.val_ratio:.0%}")
    print()

    # Generate samples first for verification
    generate_samples(args.output)

    # Generate full dataset
    metadata = generate_dataset(
        args.output,
        num_per_class=args.num_per_class,
        val_ratio=args.val_ratio,
        size=args.size,
        seed=args.seed,
    )

    print()
    print("=" * 50)
    print("Generation complete!")
    print(f"Total images: {metadata['stats']['total_generated']}")
    print(f"Train: {metadata['stats']['train_count']}")
    print(f"Val: {metadata['stats']['val_count']}")
    print(f"Output: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
