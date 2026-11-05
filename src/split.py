#!/usr/bin/env python3
"""把已标注数据(图片 + 同名 txt)按比例切分成 datasets/images + datasets/labels 的 train/val。

用法:
    python3 src/split.py --images <图片目录> --labels <标注目录>
    python3 src/split.py --images <图片目录> --labels <标注目录> --ratio 0.85 0.15
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from common import DATASETS_DIR, IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True, help="图片目录")
    ap.add_argument("--labels", type=Path, required=True, help="标注目录(与图片同名 .txt)")
    ap.add_argument("--ratio", type=float, nargs="+", default=[0.85, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = sorted(p for p in args.images.glob("*") if p.suffix.lower() in IMG_EXTS)
    images = [p for p in images if (args.labels / (p.stem + ".txt")).exists()]
    random.seed(args.seed)
    random.shuffle(images)

    n = len(images)
    tr = args.ratio[0]
    i_val = int(n * tr)
    splits = {"train": images[:i_val], "val": images[i_val:]}

    for name, lst in splits.items():
        (DATASETS_DIR / "images" / name).mkdir(parents=True, exist_ok=True)
        (DATASETS_DIR / "labels" / name).mkdir(parents=True, exist_ok=True)
        for ip in lst:
            shutil.copy2(ip, DATASETS_DIR / "images" / name / ip.name)
            shutil.copy2(args.labels / (ip.stem + ".txt"), DATASETS_DIR / "labels" / name / (ip.stem + ".txt"))
        print(f"{name}: {len(lst)} 张")

    print(f"完成 -> {DATASETS_DIR}/images/ + labels/")


if __name__ == "__main__":
    main()
