#!/usr/bin/env python3
"""
Improved synthetic data generator for printed digit recognition.

Improvements over v1:
- More diverse fonts (system fonts + downloadable)
- Realistic paper backgrounds and textures
- Better empty cell variations
- Grid line artifacts
- Perspective and lighting variations
- Hard negatives (partial digits, smudges)

Usage:
    python generate_synthetic_v2.py --output data/synthetic_v2 --num-per-class 10000
    python generate_synthetic_v2.py --preview  # Generate sample grid
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =============================================================================
# Font Discovery
# =============================================================================

# Common fonts for printed digits (prioritized by similarity to sudoku fonts)
PREFERRED_FONTS = [
    # Sans-serif (most common in sudoku)
    "Arial", "Arial Black", "Arial Narrow",
    "Helvetica", "Helvetica Neue", "Helvetica Neue Medium",
    "Verdana", "Tahoma", "Trebuchet MS",
    "Futura", "Gill Sans", "Optima",
    # Serif
    "Times New Roman", "Georgia", "Palatino",
    "Garamond", "Baskerville", "Didot",
    # Monospace (common in puzzle apps)
    "Courier New", "Courier", "Menlo", "Monaco",
    "SF Mono", "Consolas", "Lucida Console",
    # System
    "SF Pro Display", "SF Pro Text",
    ".SF NS Display", ".SF NS Text",
]

FONT_SEARCH_PATHS = [
    "/System/Library/Fonts/",
    "/Library/Fonts/",
    "~/Library/Fonts/",
    "/System/Library/Fonts/Supplemental/",
]


def find_available_fonts(size: int = 24) -> List[Tuple[str, ImageFont.FreeTypeFont]]:
    """Find all available fonts on the system."""
    found_fonts = []

    for font_name in PREFERRED_FONTS:
        font = _try_load_font(font_name, size)
        if font:
            found_fonts.append((font_name, font))

    # Also try to find fonts in search paths
    for path in FONT_SEARCH_PATHS:
        font_dir = Path(path).expanduser()
        if font_dir.exists():
            for font_file in font_dir.glob("*.ttf"):
                try:
                    font = ImageFont.truetype(str(font_file), size)
                    found_fonts.append((font_file.stem, font))
                except:
                    pass
            for font_file in font_dir.glob("*.ttc"):
                try:
                    font = ImageFont.truetype(str(font_file), size)
                    found_fonts.append((font_file.stem, font))
                except:
                    pass

    # Remove duplicates by name
    seen = set()
    unique_fonts = []
    for name, font in found_fonts:
        if name not in seen:
            seen.add(name)
            unique_fonts.append((name, font))

    return unique_fonts


def _try_load_font(font_name: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Try to load a font by name."""
    try:
        return ImageFont.truetype(font_name, size)
    except:
        pass

    # Try with extensions
    for ext in [".ttf", ".ttc", ".otf"]:
        try:
            return ImageFont.truetype(font_name + ext, size)
        except:
            pass

    # Try in font directories
    for path in FONT_SEARCH_PATHS:
        font_dir = Path(path).expanduser()
        for ext in [".ttf", ".ttc", ".otf"]:
            font_path = font_dir / (font_name + ext)
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size)
                except:
                    pass

    return None


# =============================================================================
# Background Generation
# =============================================================================

def generate_paper_background(size: int = 28) -> np.ndarray:
    """Generate realistic paper texture background."""
    variation = random.choice(["plain", "textured", "gradient", "noisy"])

    if variation == "plain":
        # Plain white/off-white
        value = random.randint(235, 255)
        img = np.full((size, size), value, dtype=np.uint8)

    elif variation == "textured":
        # Paper texture with subtle noise
        base = random.randint(240, 250)
        noise = np.random.normal(0, random.uniform(2, 5), (size, size))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)

        # Add very subtle horizontal lines (paper grain)
        if random.random() > 0.5:
            for y in range(0, size, random.randint(2, 4)):
                img[y, :] = np.clip(img[y, :].astype(np.int16) - random.randint(1, 3), 0, 255)

    elif variation == "gradient":
        # Subtle gradient (uneven lighting)
        base = random.randint(240, 250)
        img = np.full((size, size), base, dtype=np.uint8)

        direction = random.choice(["h", "v", "radial"])
        strength = random.uniform(3, 10)

        if direction == "h":
            gradient = np.linspace(-strength, strength, size)
            img = np.clip(img + gradient[np.newaxis, :], 0, 255).astype(np.uint8)
        elif direction == "v":
            gradient = np.linspace(-strength, strength, size)
            img = np.clip(img + gradient[:, np.newaxis], 0, 255).astype(np.uint8)
        else:  # radial
            y, x = np.ogrid[:size, :size]
            center = size // 2
            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            gradient = (dist / (size / 2)) * strength
            img = np.clip(img - gradient, 0, 255).astype(np.uint8)

    else:  # noisy
        base = random.randint(245, 255)
        noise = np.random.normal(0, random.uniform(3, 8), (size, size))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)

    return img


def add_grid_artifacts(img: np.ndarray) -> np.ndarray:
    """Add grid line artifacts at cell edges."""
    size = img.shape[0]
    img = img.copy()

    line_darkness = random.randint(180, 220)
    line_width = random.randint(1, 2)

    # Randomly add lines at edges
    edges = []
    if random.random() > 0.6:
        edges.append("top")
    if random.random() > 0.6:
        edges.append("bottom")
    if random.random() > 0.6:
        edges.append("left")
    if random.random() > 0.6:
        edges.append("right")

    for edge in edges:
        if edge == "top":
            img[:line_width, :] = np.minimum(img[:line_width, :], line_darkness)
        elif edge == "bottom":
            img[-line_width:, :] = np.minimum(img[-line_width:, :], line_darkness)
        elif edge == "left":
            img[:, :line_width] = np.minimum(img[:, :line_width], line_darkness)
        elif edge == "right":
            img[:, -line_width:] = np.minimum(img[:, -line_width:], line_darkness)

    return img


# =============================================================================
# Digit Generation
# =============================================================================

def generate_digit(
    digit: int,
    font: ImageFont.FreeTypeFont,
    size: int = 28,
    jitter: int = 2,
) -> np.ndarray:
    """Generate a single digit image with paper background."""
    # Start with paper background
    background = generate_paper_background(size)

    # Create digit on transparent background
    padding = 10
    img_size = size + padding * 2
    digit_img = Image.new("L", (img_size, img_size), color=255)
    draw = ImageDraw.Draw(digit_img)

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center with jitter
    x = (img_size - text_width) // 2 + random.randint(-jitter, jitter)
    y = (img_size - text_height) // 2 + random.randint(-jitter, jitter) - bbox[1]

    # Random ink darkness
    ink_color = random.randint(0, 30)
    draw.text((x, y), text, font=font, fill=ink_color)

    # Crop to target size
    digit_img = digit_img.crop((padding, padding, padding + size, padding + size))
    digit_array = np.array(digit_img)

    # Blend with background (multiply blend mode for realistic ink)
    result = (background.astype(np.float32) * digit_array.astype(np.float32) / 255).astype(np.uint8)

    return result


def generate_empty_cell(size: int = 28) -> np.ndarray:
    """Generate an empty cell with possible artifacts."""
    background = generate_paper_background(size)

    # Sometimes add grid artifacts
    if random.random() > 0.5:
        background = add_grid_artifacts(background)

    # Sometimes add very faint marks/smudges
    if random.random() > 0.8:
        # Add a small smudge
        smudge_size = random.randint(2, 5)
        x = random.randint(0, size - smudge_size)
        y = random.randint(0, size - smudge_size)
        smudge_darkness = random.randint(210, 240)
        background[y:y+smudge_size, x:x+smudge_size] = np.minimum(
            background[y:y+smudge_size, x:x+smudge_size],
            smudge_darkness
        )

    return background


# =============================================================================
# Augmentation
# =============================================================================

def apply_augmentation(img: np.ndarray, is_empty: bool = False) -> np.ndarray:
    """Apply random augmentations to simulate real-world conditions."""
    h, w = img.shape

    # Rotation
    if random.random() > 0.3:
        angle = random.uniform(-8, 8) if is_empty else random.uniform(-10, 10)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=255)

    # Scale
    if random.random() > 0.4:
        scale = random.uniform(0.9, 1.1)
        new_size = max(1, int(w * scale))
        scaled = cv2.resize(img, (new_size, new_size))

        if new_size > w:
            start = (new_size - w) // 2
            img = scaled[start:start+h, start:start+w]
        else:
            pad = (w - new_size) // 2
            img = cv2.copyMakeBorder(scaled, pad, w-new_size-pad,
                                    pad, w-new_size-pad,
                                    cv2.BORDER_CONSTANT, value=255)
            if img.shape[0] != h:
                img = cv2.resize(img, (w, h))

    # Blur (focus issues)
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.3, 1.0)
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # Brightness
    if random.random() > 0.4:
        brightness = random.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

    # Contrast
    if random.random() > 0.4:
        contrast = random.uniform(0.85, 1.15)
        mean = np.mean(img)
        img = np.clip((img.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # Noise
    if random.random() > 0.5:
        noise_sigma = random.uniform(2, 8)
        noise = np.random.normal(0, noise_sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Perspective
    if random.random() > 0.7:
        strength = random.uniform(1, 2.5)
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([
            [random.uniform(0, strength), random.uniform(0, strength)],
            [w - random.uniform(0, strength), random.uniform(0, strength)],
            [w - random.uniform(0, strength), h - random.uniform(0, strength)],
            [random.uniform(0, strength), h - random.uniform(0, strength)]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matrix, (w, h), borderValue=255)

    # Erosion/Dilation (ink spread)
    if not is_empty and random.random() > 0.8:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    return img


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_dataset(
    output_dir: Path,
    num_per_class: int = 10000,
    val_ratio: float = 0.1,
    size: int = 28,
    seed: int = 42,
) -> dict:
    """Generate the full synthetic dataset."""
    random.seed(seed)
    np.random.seed(seed)

    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    for split_dir in [train_dir, val_dir]:
        for digit in range(10):
            (split_dir / str(digit)).mkdir(parents=True, exist_ok=True)

    # Find fonts
    fonts = find_available_fonts(size=random.randint(18, 24))
    print(f"Found {len(fonts)} fonts")

    if not fonts:
        print("Warning: No fonts found, using default")
        fonts = [("default", ImageFont.load_default())]

    # Statistics
    stats = {
        "total_generated": 0,
        "train_count": 0,
        "val_count": 0,
        "per_class": {str(d): {"train": 0, "val": 0} for d in range(10)},
        "fonts_used": [f[0] for f in fonts[:20]],  # First 20 font names
    }

    num_val = int(num_per_class * val_ratio)
    num_train = num_per_class - num_val

    for digit in range(10):
        print(f"Generating class {digit}...", end=" ", flush=True)

        for i in range(num_per_class):
            is_val = i < num_val
            split_dir = val_dir if is_val else train_dir
            split_name = "val" if is_val else "train"

            # Generate image
            if digit == 0:
                img = generate_empty_cell(size=size)
                img = apply_augmentation(img, is_empty=True)
            else:
                font_name, font = random.choice(fonts)
                # Reload font with random size
                font_size = random.randint(16, 24)
                font = _try_load_font(font_name, font_size) or font
                img = generate_digit(digit, font, size=size, jitter=2)
                img = apply_augmentation(img, is_empty=False)

            # Ensure correct size
            if img.shape != (size, size):
                img = cv2.resize(img, (size, size))

            # Save
            filename = f"{digit}_{i:06d}.png"
            cv2.imwrite(str(split_dir / str(digit) / filename), img)

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


def generate_preview(output_path: Path, size: int = 28):
    """Generate a preview grid showing sample images from each class."""
    fonts = find_available_fonts(size=22)
    if not fonts:
        fonts = [("default", ImageFont.load_default())]

    # Create 10x5 grid (each class x 5 samples)
    grid_rows = 10
    grid_cols = 5
    cell_size = size * 2  # Larger for visibility
    padding = 4

    grid_h = grid_rows * (cell_size + padding) + padding
    grid_w = grid_cols * (cell_size + padding) + padding
    grid = np.full((grid_h, grid_w), 200, dtype=np.uint8)

    for digit in range(10):
        for sample in range(grid_cols):
            # Generate sample
            if digit == 0:
                img = generate_empty_cell(size=size)
            else:
                font_name, font = random.choice(fonts)
                img = generate_digit(digit, font, size=size)

            # Apply light augmentation for variety
            img = apply_augmentation(img, is_empty=(digit == 0))

            # Resize for visibility
            img = cv2.resize(img, (cell_size, cell_size))

            # Place in grid
            y = digit * (cell_size + padding) + padding
            x = sample * (cell_size + padding) + padding
            grid[y:y+cell_size, x:x+cell_size] = img

    cv2.imwrite(str(output_path), grid)
    print(f"Preview saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic digit dataset (v2)")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "synthetic_v2",
        help="Output directory",
    )
    parser.add_argument(
        "--num-per-class", "-n",
        type=int,
        default=10000,
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
        "--preview",
        action="store_true",
        help="Only generate preview image",
    )
    args = parser.parse_args()

    print(f"Output directory: {args.output}")

    if args.preview:
        args.output.mkdir(parents=True, exist_ok=True)
        generate_preview(args.output / "preview.png", args.size)
        return

    print(f"Generating {args.num_per_class} images per class (10 classes)")
    print(f"Total images: {args.num_per_class * 10}")
    print(f"Train/val split: {1 - args.val_ratio:.0%}/{args.val_ratio:.0%}")
    print()

    metadata = generate_dataset(
        args.output,
        num_per_class=args.num_per_class,
        val_ratio=args.val_ratio,
        size=args.size,
        seed=args.seed,
    )

    # Generate preview
    generate_preview(args.output / "preview.png", args.size)

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
