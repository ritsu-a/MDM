## MotionStat300（motion_stat_300）训练与推理

本文档只描述本仓库对 **`motion_stat_300`** 数据集的训练、推理与可视化流程。

### 环境准备

- **conda**：建议使用你已有的 `mdm` 环境跑训练/采样
- **依赖**：
  - `ffmpeg`（如需生成视频）
  - `spacy` + 英文模型（用于文本分词）
  - 词向量：先执行 `prepare/download_glove.sh`，生成 `./glove/our_vab_*`

### 数据组织格式（data/motion_stat_300）

数据根目录默认是 `data/motion_stat_300`，也可以用 `--data_dir` 指定。

目录内需要至少包含：

- **划分文件**
  - `train.txt`
  - `val.txt`
  -（可选）`test.txt`
  - 每行一个样本 ID（例如：`ARM_ENDEAVOR-1`）

- **文本描述**
  - `{ID}.txt`
  - 文件内容是一段英文描述（整段作为 caption）

- **动作**
  - `{ID}_motion.npz`
  - 优先读取键 **`qpos`**（其次 `motion`，否则取 npz 内第一个数组）
  - 形状 **(300, 60)**：300 帧、每帧 60 维 qpos

首次训练/推理若目录下没有 `Mean.npy / Std.npy`，会基于 `train.txt` 自动统计并写入：

- `data/motion_stat_300/Mean.npy`
- `data/motion_stat_300/Std.npy`

### 训练（mdm 环境）

默认数据目录：

```bash
conda activate mdm
CUDA_VISIBLE_DEVICES=1 python -m train.train_mdm \
  --save_dir save/motion_stat_300_t2m \
  --dataset motion_stat_300 --save_interval 10000 \
  --overwrite
```

如果数据不在默认目录，显式指定：

```bash
conda activate mdm
CUDA_VISIBLE_DEVICES=1 python -m train.train_mdm \
  --save_dir save/my_motion_stat_300 \
  --dataset motion_stat_300 \
  --data_dir /root/workspace/motion-diffusion-model/data/motion_stat_300
```

### 推理：生成 qpos npz（mdm 环境）

`motion_stat_300` 的 fps 按 **60** 处理。要生成 **300 帧**，需要：

- \(300 / 60 = 5\) 秒，因此使用 `--motion_length 5`

示例（保存每条样本为 `.npz`，键名 `qpos`，形状 `(300,60)`）：

```bash
conda activate mdm
CUDA_VISIBLE_DEVICES=2 python -m sample.generate \
  --model_path ./save/my_motion_stat_300/model000100000.pt \
  --dataset motion_stat_300 \
  --text_prompt "A person raises one hand." \
  --motion_length 5 \
  --save_npz
```

输出目录会包含：

- `results.npy`（汇总）
- `npz/sample0000.npz`、`npz/sample0001.npz` ...

### 可视化：用 MARDM 环境渲染视频（推荐分离流程）

推荐流程是：

- **mdm 环境**：只负责生成 `npz/`（不要 `--render_video`）
- **MARDM 环境**：用渲染脚本把 `.npz` 转成 `.mp4`

渲染单个样本示例（按你的脚本路径）：

```bash
conda activate MARDM
python /root/workspace/MotionDiT/external/GMR/scripts/vis_npz_motion.py \
  --npz_path /root/workspace/motion-diffusion-model/save/my_motion_stat_300/samples_*/npz/sample0000.npz \
  --video_path /root/workspace/motion-diffusion-model/save/my_motion_stat_300/samples_*/videos/sample0000.mp4 \
  --motion_fps 60
```

> 你也可以在 `mdm` 采样时加 `--render_video --render_script_path ...` 让它自动渲染；但如果你希望严格分离环境（mdm 只产 npz，MARDM 才渲染），就不要在 mdm 侧加 `--render_video`。

