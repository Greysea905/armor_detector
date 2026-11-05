"""可复用模块:路径解析 + 配置/模型加载"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data, path):
    """把 dict 写成 yaml 文件(自动建父目录)"""
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_model(weights):
    """加载 YOLO 模型。weights 为 .pt 路径或模型名(找不到会在线下载)"""
    from ultralytics import YOLO
    return YOLO(str(weights))
