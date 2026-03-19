# G1 BrainCo Hands DoF 结构分析（60 = 7 root + 53 joints）

- 源文件：`external/GMR/assets/g1_brainco/G1_brainco_hands.urdf`
- 总状态维度口径：**60 DoF = 7(root) + 53(revolute joints)**
- 其中 root 7 维：`[x, y, z, qw, qx, qy, qz]`（位置 3 + 旋转四元数 4；MuJoCo viewer 使用 wxyz）
- 总 revolute 关节数（按 URDF 直接计数）：**53**
- 其中主躯干+四肢+腕部：**29**（29 DoF）
- 左手（含 mimic 与 tip 微关节）：**12**
- 右手（含 mimic 与 tip 微关节）：**12**
- 合计：**29 + 24 = 53 DoF**

## 1) DoF 组成（URDF 实测）

| 模块 | DoF | 说明 |
|---|---:|---|
| 左腿 | 6 | hip pitch/roll/yaw + knee + ankle pitch/roll |
| 右腿 | 6 | hip pitch/roll/yaw + knee + ankle pitch/roll |
| 腰部 | 3 | yaw/roll/pitch |
| 左臂 | 7 | shoulder(3) + elbow(1) + wrist(3) |
| 右臂 | 7 | shoulder(3) + elbow(1) + wrist(3) |
| 左手 | 12 | 拇指4 + 其余四指各2（带 mimic） + tip 微关节 |
| 右手 | 12 | 拇指4 + 其余四指各2（带 mimic） + tip 微关节 |
| **总计** | **53** | **URDF 中可见的 revolute 关节数** |

## 2) 60 DoF 口径说明

- 本项目中的“60 DoF”采用**状态向量维度**口径，而非纯机械关节数口径：
- `60 = 7(root) + 53(revolute joints)`。
- 其中 root 使用 7 维表示：位置 `x,y,z` + 姿态四元数 `qw,qx,qy,qz`（wxyz）。
- 结论：**URDF 机械链 revolute 关节数是 53；加上 root 7 维后为 60。**

## 3) mimic 与“独立控制 DoF”

- revolute 中带 `<mimic>` 的关节数：**10**（左右手各 5 个：thumb_distal + 四指 distal）
- 非 mimic 的 revolute 关节数：**43**
- 另外，左右拇指 tip 关节虽为 revolute，但限位为 `[-0.001, 0.001]` 且 `effort=0, velocity=0`，通常可视作几何补偿/被动微关节。
- 若将 mimic 与 tip 微关节都视为非独立执行器，则常见“可独立驱动 DoF”约为：**41**。

## 4) 轴向分布（53 个 revolute）

| 关节轴 | 数量 |
|---|---:|
| `-1 0 0` | 1 |
| `0 0 1` | 9 |
| `0 1 0` | 29 |
| `1 0 0` | 14 |

## 5) 全部 53 个 revolute 关节明细

| # | joint | parent -> child | axis | lower | upper | effort | velocity | mimic |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | `left_hip_pitch_joint` | `pelvis` -> `left_hip_pitch_link` | `0 1 0` | -2.5307 | 2.8798 | 88 | 32 | - |
| 2 | `left_hip_roll_joint` | `left_hip_pitch_link` -> `left_hip_roll_link` | `1 0 0` | -0.5236 | 2.9671 | 139 | 20 | - |
| 3 | `left_hip_yaw_joint` | `left_hip_roll_link` -> `left_hip_yaw_link` | `0 0 1` | -2.7576 | 2.7576 | 88 | 32 | - |
| 4 | `left_knee_joint` | `left_hip_yaw_link` -> `left_knee_link` | `0 1 0` | -0.087267 | 2.8798 | 139 | 20 | - |
| 5 | `left_ankle_pitch_joint` | `left_knee_link` -> `left_ankle_pitch_link` | `0 1 0` | -0.87267 | 0.5236 | 35 | 30 | - |
| 6 | `left_ankle_roll_joint` | `left_ankle_pitch_link` -> `left_ankle_roll_link` | `1 0 0` | -0.2618 | 0.2618 | 35 | 30 | - |
| 7 | `right_hip_pitch_joint` | `pelvis` -> `right_hip_pitch_link` | `0 1 0` | -2.5307 | 2.8798 | 88 | 32 | - |
| 8 | `right_hip_roll_joint` | `right_hip_pitch_link` -> `right_hip_roll_link` | `1 0 0` | -2.9671 | 0.5236 | 139 | 20 | - |
| 9 | `right_hip_yaw_joint` | `right_hip_roll_link` -> `right_hip_yaw_link` | `0 0 1` | -2.7576 | 2.7576 | 88 | 32 | - |
| 10 | `right_knee_joint` | `right_hip_yaw_link` -> `right_knee_link` | `0 1 0` | -0.087267 | 2.8798 | 139 | 20 | - |
| 11 | `right_ankle_pitch_joint` | `right_knee_link` -> `right_ankle_pitch_link` | `0 1 0` | -0.87267 | 0.5236 | 35 | 30 | - |
| 12 | `right_ankle_roll_joint` | `right_ankle_pitch_link` -> `right_ankle_roll_link` | `1 0 0` | -0.2618 | 0.2618 | 35 | 30 | - |
| 13 | `waist_yaw_joint` | `pelvis` -> `waist_yaw_link` | `0 0 1` | -2.618 | 2.618 | 88 | 32 | - |
| 14 | `waist_roll_joint` | `waist_yaw_link` -> `waist_roll_link` | `1 0 0` | -0.52 | 0.52 | 35 | 30 | - |
| 15 | `waist_pitch_joint` | `waist_roll_link` -> `torso_link` | `0 1 0` | -0.52 | 0.52 | 35 | 30 | - |
| 16 | `left_shoulder_pitch_joint` | `torso_link` -> `left_shoulder_pitch_link` | `0 1 0` | -3.0892 | 2.6704 | 25 | 30 | - |
| 17 | `left_shoulder_roll_joint` | `left_shoulder_pitch_link` -> `left_shoulder_roll_link` | `1 0 0` | -1.5882 | 2.2515 | 25 | 37 | - |
| 18 | `left_shoulder_yaw_joint` | `left_shoulder_roll_link` -> `left_shoulder_yaw_link` | `0 0 1` | -2.618 | 2.618 | 25 | 37 | - |
| 19 | `left_elbow_joint` | `left_shoulder_yaw_link` -> `left_elbow_link` | `0 1 0` | -1.0472 | 2.0944 | 25 | 37 | - |
| 20 | `left_wrist_roll_joint` | `left_elbow_link` -> `left_wrist_roll_link` | `1 0 0` | -1.972222054 | 1.972222054 | 25 | 37 | - |
| 21 | `left_wrist_pitch_joint` | `left_wrist_roll_link` -> `left_wrist_pitch_link` | `0 1 0` | -1.614429558 | 1.614429558 | 5 | 22 | - |
| 22 | `left_wrist_yaw_joint` | `left_wrist_pitch_link` -> `left_wrist_yaw_link` | `0 0 1` | -1.614429558 | 1.614429558 | 5 | 22 | - |
| 23 | `right_shoulder_pitch_joint` | `torso_link` -> `right_shoulder_pitch_link` | `0 1 0` | -3.0892 | 2.6704 | 25 | 37 | - |
| 24 | `right_shoulder_roll_joint` | `right_shoulder_pitch_link` -> `right_shoulder_roll_link` | `1 0 0` | -2.2515 | 1.5882 | 25 | 37 | - |
| 25 | `right_shoulder_yaw_joint` | `right_shoulder_roll_link` -> `right_shoulder_yaw_link` | `0 0 1` | -2.618 | 2.618 | 25 | 37 | - |
| 26 | `right_elbow_joint` | `right_shoulder_yaw_link` -> `right_elbow_link` | `0 1 0` | -1.0472 | 2.0944 | 25 | 37 | - |
| 27 | `right_wrist_roll_joint` | `right_elbow_link` -> `right_wrist_roll_link` | `1 0 0` | -1.972222054 | 1.972222054 | 25 | 37 | - |
| 28 | `right_wrist_pitch_joint` | `right_wrist_roll_link` -> `right_wrist_pitch_link` | `0 1 0` | -1.614429558 | 1.614429558 | 5 | 22 | - |
| 29 | `right_wrist_yaw_joint` | `right_wrist_pitch_link` -> `right_wrist_yaw_link` | `0 0 1` | -1.614429558 | 1.614429558 | 5 | 22 | - |
| 30 | `left_thumb_metacarpal_joint` | `left_base_link` -> `left_thumb_metacarpal_Link` | `0 0 1` | 0 | 1.5184 | 0.5 | 2.6175 | - |
| 31 | `left_thumb_proximal_joint` | `left_thumb_metacarpal_Link` -> `left_thumb_proximal_Link` | `1 0 0` | 0 | 1.0472 | 1.1 | 2.5303 | - |
| 32 | `left_thumb_distal_joint` | `left_thumb_proximal_Link` -> `left_thumb_distal_Link` | `1 0 0` | 0 | 1.0472 | 1.1 | 2.5303 | left_thumb_proximal_joint |
| 33 | `left_thumb_tip_joint` | `left_thumb_distal_Link` -> `left_thumb_tip_Link` | `-1 0 0` | -0.001 | 0.001 | 0 | 0 | - |
| 34 | `left_index_proximal_joint` | `left_base_link` -> `left_index_proximal_Link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 35 | `left_index_distal_joint` | `left_index_proximal_Link` -> `left_index_distal_Link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | left_index_proximal_joint |
| 36 | `left_middle_proximal_joint` | `left_base_link` -> `left_middle_proximal_Link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 37 | `left_middle_distal_joint` | `left_middle_proximal_Link` -> `left_middle_distal_Link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | left_middle_proximal_joint |
| 38 | `left_ring_proximal_joint` | `left_base_link` -> `left_ring_proximal_Link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 39 | `left_ring_distal_joint` | `left_ring_proximal_Link` -> `left_ring_distal_Link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | left_ring_proximal_joint |
| 40 | `left_pinky_proximal_joint` | `left_base_link` -> `left_pinky_proximal_Link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 41 | `left_pinky_distal_joint` | `left_pinky_proximal_Link` -> `left_pinky_distal_Link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | left_pinky_proximal_joint |
| 42 | `right_thumb_metacarpal_joint` | `right_base_link` -> `right_thumb_metacarpal_link` | `0 0 1` | 0 | 1.5184 | 0.5 | 2.6175 | - |
| 43 | `right_thumb_proximal_joint` | `right_thumb_metacarpal_link` -> `right_thumb_proximal_link` | `1 0 0` | 0 | 1.0472 | 1.1 | 2.5303 | - |
| 44 | `right_thumb_distal_joint` | `right_thumb_proximal_link` -> `right_thumb_distal_link` | `1 0 0` | 0 | 1.0472 | 1.1 | 2.5303 | right_thumb_proximal_joint |
| 45 | `right_thumb_tip_joint` | `right_thumb_distal_link` -> `right_thumb_tip` | `1 0 0` | -0.001 | 0.001 | 0 | 0 | - |
| 46 | `right_index_proximal_joint` | `right_base_link` -> `right_index_proximal_link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 47 | `right_index_distal_joint` | `right_index_proximal_link` -> `right_index_distal_link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | right_index_proximal_joint |
| 48 | `right_middle_proximal_joint` | `right_base_link` -> `right_middle_proximal_link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 49 | `right_middle_distal_joint` | `right_middle_proximal_link` -> `right_middle_distal_link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | right_middle_proximal_joint |
| 50 | `right_ring_proximal_joint` | `right_base_link` -> `right_ring_proximal_link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 51 | `right_ring_distal_joint` | `right_ring_proximal_link` -> `right_ring_distal_link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | right_ring_proximal_joint |
| 52 | `right_pinky_proximal_joint` | `right_base_link` -> `right_pinky_proximal_link` | `0 1 0` | 0 | 1.4661 | 2 | 2.2685 | - |
| 53 | `right_pinky_distal_joint` | `right_pinky_proximal_link` -> `right_pinky_distal_link` | `0 1 0` | 0 | 1.693 | 2 | 2.2685 | right_pinky_proximal_joint |