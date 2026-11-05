#!/usr/bin/env python3
"""预标注脚本：视频抽帧 + 当前模型推理 + 输出 labelimg 可读的 YOLO txt。

用途：半监督优化 bbox。用当前模型（线上 onnx，288）对录像抽帧预标注，
      labelimg 人工检查/修正后，并入训练集加训。

用法：
  python3 src/prelabel.py /path/to/video.avi \
      --interval 20 --out prelabel \
      --model detect/models/armor_detector.onnx --imgsz 288

参数：
  --interval  抽帧间隔（默认 20；相邻帧高度相似，可加大到 30~50 去重）
  --out       输出目录
  --model     模型路径（.onnx 或 .pt；线上 onnx 用 imgsz=288）
  --imgsz     推理尺寸（默认 288，和线上一致；若用 416 的 .pt 则改 416）
  --conf      置信度阈值

输出（同目录，labelimg 直接打开 <out>/ 即可）：
  <out>/xxxxxx.jpg   抽帧图片
  <out>/xxxxxx.txt   YOLO txt：每行 "class_id cx cy w h"（归一化）
  <out>/classes.txt  labelimg 用的类别文件（dead/red/blue）

注意：class_id 与 config/data.yaml 一致（0=dead, 1=red, 2=blue），
      脚本只保留 red/blue，丢掉 dead，可直接并入训练集。
"""

import argparse
import os

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="视频路径")
    ap.add_argument("--interval", type=int, default=20, help="抽帧间隔")
    ap.add_argument("--out", default="prelabel", help="输出目录")
    ap.add_argument("--model", default="detect/models/armor_detector.onnx",
                    help="模型路径(.onnx/.pt)")
    ap.add_argument("--imgsz", type=int, default=288, help="推理尺寸")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--no-flip", action="store_true", help="不翻转（视频不倒置时用）")
    ap.add_argument("--keep", default="1,2", help="保留的类别，逗号分隔（默认 1,2；只补蓝色用 --keep 2）")
    args = ap.parse_args()

    keep = set(int(x) for x in args.keep.split(","))

    model = YOLO(args.model)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("dead\nred\nblue\n")

    cap = cv2.VideoCapture(args.video)
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.interval != 0:
            idx += 1
            continue

        if not args.no_flip:
            frame = cv2.flip(frame, -1)  # 相机倒装，180° 转正后再存图/推理

        cv2.imwrite(os.path.join(args.out, f"{saved:06d}.jpg"), frame)

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        with open(os.path.join(args.out, f"{saved:06d}.txt"), "w") as f:
            for b in res.boxes:
                cls = int(b.cls[0])
                if cls not in keep:
                    continue  # 只保留 --keep 指定的类别
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                h, w = frame.shape[:2]
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved += 1
        idx += 1

    cap.release()
    print(f"抽帧 {saved} 张 → {args.out}/ (jpg + txt + classes.txt)")


if __name__ == "__main__":
    main()
