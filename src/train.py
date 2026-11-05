#!/usr/bin/env python3
"""训练入口:读 config/train.yaml 里的超参,调用 ultralytics 训练并导出 ONNX。

用法(项目根目录运行):
    python3 src/train.py                    # 用 train.yaml 默认参数
    python3 src/train.py --device 0         # AutoDL GPU(默认已是 0)
    python3 src/train.py --epochs 200 --batch 32 --imgsz 1280
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, DATASETS_DIR, CONFIG_DIR, RUNS_DIR, load_yaml, dump_yaml, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, default=CONFIG_DIR / "train.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    for key in ("epochs", "imgsz", "batch", "device"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v

    # 把 data.yaml 的 train/val/names 抽出来,配上 datasets/ 绝对路径,生成临时数据配置
    dcfg = load_yaml(PROJECT_ROOT / cfg["data"])
    data_tmp = RUNS_DIR / "_data.yaml"
    dump_yaml(
        {"path": str(DATASETS_DIR), "train": dcfg["train"], "val": dcfg["val"], "names": dcfg["names"]},
        data_tmp,
    )

    model = load_model(PROJECT_ROOT / cfg["model"])

    train_args = {
        "data": str(data_tmp),
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "device": cfg["device"],
        "workers": cfg.get("workers", 8),
        "project": str(RUNS_DIR),
        "name": cfg.get("name", "armor"),
        "exist_ok": False,
    }
    for key in ("lr0", "lrf", "cos_lr", "warmup_epochs", "patience", "save_period", "close_mosaic",
                "mosaic", "optimizer", "weight_decay", "hsv_h", "hsv_s", "hsv_v", "fliplr"):
        if key in cfg:
            train_args[key] = cfg[key]

    model.train(**train_args)

    # 导出 ONNX(供 C++ 推理),与训练 imgsz 一致
    best = RUNS_DIR / cfg.get("name", "armor") / "weights" / "best.pt"
    load_model(best).export(format="onnx", imgsz=cfg["imgsz"])
    print(f"\n导出完成:{best.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
