# armor_detection — 装甲板检测模型训练

训练 RoboMaster 装甲板检测模型(YOLOv26n),产出 ONNX 权重,交给 C++ 主工程 `rmv_autoaim_lxy/detect/` 做实时推理。

## 1. 数据来源

[ENTERPRIZE 战队 2025 开源数据集](https://github.com/hkustenterprize/RM2025-Radar-Algorithm)(MIT 协议)

- **装甲板识别数据:3504 张**,标签:
  - `0` = 死亡状态装甲板
  - `1` = 红色灯条装甲板
  - `2` = 蓝色灯条装甲板
- 下载:SharePoint 链接(密码 `987654321`),见 ENTERPRIZE 仓库 README「训练指南」。

> 引用规范:报告里注明数据来源(ENTERPRIZE 战队 2025、链接、MIT)。

## 2. 目录结构

```
armor_detection/
├── config/
│   ├── data.yaml      # 数据配置(类别 0/1/2)
│   └── train.yaml     # 训练超参(模型/imgsz/epochs/学习率)
├── src/
│   ├── common.py      # 路径解析 + 配置/模型加载
│   ├── train.py       # 训练入口 + 导出 ONNX
│   ├── val.py         # 验证集评估(mAP50/mAP50-95)
│   └── split.py       # 划分 train/val
├── datasets/          # 切分后的数据(images/labels 的 train/val)
├── weights/           # 预训练权重 yolo26n.pt(首次自动下载)
├── runs/              # 训练输出(gitignored)
└── requirements.txt
```

## 3. 快速开始(云端 AutoDL)

### 3.1 下载数据

从 SharePoint 下载 ENTERPRIZE 装甲板识别数据(3504 张,密码 `987654321`)。

### 3.2 整理并切分

下载解压后,把图片和同名标注切分成 `datasets/`:

```bash
python3 src/split.py --images <图片目录> --labels <标注目录> --ratio 0.85 0.15
```

> 若数据已是 train/val 结构,直接摆成 `datasets/images/{train,val}` + `datasets/labels/{train,val}` 即可,跳过 split。

### 3.3 训练

```bash
python3 src/train.py                        # 用 train.yaml 默认(device 0 GPU)
python3 src/train.py --epochs 200 --batch 32
```

训练输出到 `runs/armor/`,最优权重 `runs/armor/weights/best.pt`,并自动导出 `best.onnx`。

### 3.4 验证

```bash
python3 src/val.py --weights runs/armor/weights/best.pt --device 0
# 输出 mAP50 / mAP50-95
```

### 3.5 交给 C++

把 `runs/armor/weights/best.onnx` 复制到:
```
rmv_autoaim_lxy/detect/models/armor_yolo.onnx
```

## 4. 关键参数(train.yaml)

| 参数 | 值 | 说明 |
|---|---|---|
| model | `weights/yolo26n.pt` | 预训练权重 |
| imgsz | `1280` | 全帧检测,装甲板远处小 |
| epochs | `100` | 有早停(patience 15) |
| batch | `16` | RTX 4090D 24GB |
| device | `0` | AutoDL GPU |

> ⚠️ yolo26 较新,导出 ONNX 若在 OpenCV `cv::dnn` 里加载失败(新算子),改用 ONNX Runtime 推理,或退回 `yolo11n`。
