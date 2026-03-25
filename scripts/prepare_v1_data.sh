#!/usr/bin/env bash

set -euo pipefail

V1_DATA_ROOT="${V1_DATA_ROOT:-/project/v1_data}"
V1_DATA_URL="${V1_DATA_URL:-https://ndownloader.figshare.com/files/40805201}"
ZIP_PATH="${ZIP_PATH:-${V1_DATA_ROOT}.zip}"

mkdir -p "$(dirname "$V1_DATA_ROOT")"

echo "Downloading V1 dataset to $ZIP_PATH"
wget -O "$ZIP_PATH" "$V1_DATA_URL"

echo "Unzipping dataset into $(dirname "$V1_DATA_ROOT")"
unzip -o "$ZIP_PATH" -d "$(dirname "$V1_DATA_ROOT")"

echo "Converting PNG images to NPY files in $V1_DATA_ROOT/images_npy"
V1_DATA_ROOT="$V1_DATA_ROOT" python3 - <<'PY'
from pathlib import Path
import os

import numpy as np
from PIL import Image

data_root = Path(os.environ["V1_DATA_ROOT"])
src_dir = data_root / "images"
dst_dir = data_root / "images_npy"

if not src_dir.exists():
    raise FileNotFoundError(
        f"Missing image directory: {src_dir}. "
        "Check that the archive unpacked into the expected V1_DATA_ROOT."
    )

dst_dir.mkdir(parents=True, exist_ok=True)

count = 0
for png_path in sorted(src_dir.glob("*.png")):
    img = Image.open(png_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    np.save(dst_dir / f"{png_path.stem}.npy", arr, allow_pickle=False)
    count += 1

print(f"Saved {count} .npy files to {dst_dir}")
PY

echo "Done. Set V1_DATA_ROOT=$V1_DATA_ROOT when running analysis scripts if needed."
