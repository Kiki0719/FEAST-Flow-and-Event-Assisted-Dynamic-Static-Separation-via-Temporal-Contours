# FEAST — Flow and Event-Assisted Dynamic-Static Separation via Temporal Contours


FEAST 是 MERCY 中面向**单目固定视角动态物体**的运动区域发现、运动验证与动态 Gaussian 绑定模块。以 Canonical 3D Gaussians、初始 Gaussian–Part Binding、单目固定相机拍摄的视频和事件流为基础，识别动态部件并输出最终 Dynamic Gaussian Binding。

## FEAST 的两个版本

| 版本 | 核心方法 | 主要定位 |
| --- | --- | --- |
| **FEAST V1 — CoTracker-based** | **CoTracker3** | 利用稀疏长时序点轨迹识别和验证动态部件，形成动态 Gaussian Binding；运动覆盖受点采样影响 |
| **FEAST V2 — WAFT + CoTracker3 + Event** | **WAFT 稠密光流 + CoTracker3 点轨迹 + Event 事件证据** | WAFT 主导运动区域发现；CoTracker3 验证跨帧轨迹；Event 辅助检查遗漏区域；最终形成 Dynamic Gaussian Binding |

**版本演进：** V1 以 CoTracker3 为核心获取二维运动轨迹；V2 引入 WAFT 提升对运动区域的覆盖，并加入事件轮廓作为辅助验证。**事件不直接提供 3D 运动真值，CoTracker3 的稀疏轨迹也不应替代 WAFT 的稠密运动发现。**

### FEAST V1 流程

```text
Canonical G0 + Part IDs + 固定相机 + RGB 视频
                         ↓
                CoTracker3 点跟踪
                         ↓
              运动部件识别与验证
                         ↓
              Dynamic Gaussian Binding
```

### FEAST V2 流程

```text
Canonical G0 + Part IDs + 固定相机 + RGB + Event
                         ↓
                 WAFT — Dense Optical Flow
                         ↓
             Stage A | 2D Motion Discovery
                         ↓
             Stage B | CoTracker3 + Event 验证
                         ↓
             Stage C | Dynamic Gaussian Binding
```

### V2 的方法分工

1. **WAFT — Dense Optical Flow：** 计算连续 RGB 帧之间的稠密光流 $F_{t\to t+1}$，以运动证据发现候选动态区域和对应 Part。
2. **CoTracker3 — Point Tracking：** 提供跨帧稀疏轨迹，辅助确认候选区域的时序运动一致性；不能因缺乏跟踪点就把区域认定为静态。
3. **Event — Motion Evidence：** 利用事件轮廓与时空活动检查 WAFT 可能遗漏的运动区域，作为辅助证据而不是直接运动真值。
4. **Dynamic Gaussian Binding：** 将光流候选与运动验证结果映射回 Canonical G0，输出动态 Gaussian 索引及 Part 归属。

**命令适配说明：** 以下保留至 Dynamic Gaussian Binding 的执行步骤。文档将光流统一标注为 WAFT；Python 模块入口已改为 `FEAST.*`，对应目录为 `/root/autodl-tmp/MERCY/FEAST`。命令能否执行仍取决于该目录下实际脚本名与参数接口；尤其需要检查 `generate_waft_flow.py` 是否存在。

## 1. 输入数据与配置
### 1.1 输入结构
```text

/root/autodl-tmp/

├── Mono_Data/

│   └── <MODEL>/

│       ├── corrected_g0.ply          # Canonical Gaussians；N 个 Gaussian

│       ├── <VIDEO>.mkv               # 与光流、事件对应的 RGB 视频；T 帧

│       ├── <MODEL>_event.npy         # 异步事件流

│       ├── fixed_camera.json         # MERCY/Stage 1 固定相机

│       └── <MODEL>.blend             # 可选：Blender 部件几何入口

└── MERCY/

    ├── FEAST/                         # FEAST Python 模块及本文命令入口

    ├── scripts/motion/             # MERCY 部件提取 / 绑定脚本

    └── outputs/

        ├── \<model>-init/

        │   └── initial-binding.npz  # labels: (N,)；SAGA 或 Blender 几何配准产生

        └── \<model>-inspection/

            └── inspection.json     # camera.pose、width、height、fovy_degrees

```



`labels[i]` 必须对应 `corrected_g0.ply` 中第 `i` 个 Gaussian。`initial-binding.npz` 的 `dynamic_indices` 可能来自 Blender 动画，只用于旧接口兼容；FEAST 的最终动态部件由 Stage A/B 重新判定。



### 1.2 通用配置
```bash
conda activate monoevent
cd /root/autodl-tmp/MERCY/FEAST
export FEAST_ROOT="$PWD"
export MERCY_ROOT="$(dirname "$FEAST_ROOT")"
export PYTHONPATH="$MERCY_ROOT:$PYTHONPATH"

export MODEL="<MODEL>"
export DATA="/path/to/Mono_Data/$MODEL"
export VIDEO="$DATA/<VIDEO>.mkv"
export EVENTS="$DATA/${MODEL}_event.npy"
export G0="$DATA/corrected_g0.ply"
export FRAMES=100  # 示例值：按实际视频帧数修改
export WIDTH=640 HEIGHT=640

export INIT="$MERCY_ROOT/outputs/$MODEL-init"
export CAMERA="$MERCY_ROOT/outputs/$MODEL-inspection/inspection.json"
export BINDING="$INIT/initial-binding.npz"
export RUN="$MERCY_ROOT/outputs/$MODEL/feast_motion_discovery"

export WAFT_REPO=/root/autodl-tmp/WAFT
export WAFT_CFG=/path/to/matching-waft-config.json
export WAFT_CKPT=/path/to/matching-waft-model.pth
export COTRACKER_CKPT=/root/autodl-tmp/co-tracker/checkpoints/scaled_offline.pth
mkdir -p "$INIT" "$RUN"
```



### 1.3 Fan 示例配置
```bash
conda activate monoevent
cd /root/autodl-tmp/MERCY/FEAST
export FEAST_ROOT="$PWD"
export MERCY_ROOT="$(dirname "$FEAST_ROOT")"
export PYTHONPATH="$MERCY_ROOT:$PYTHONPATH"

export MODEL=Fan
export DATA=/root/autodl-tmp/Mono_Data/Fan
export VIDEO="$DATA/Fan.mkv"
export EVENTS="$DATA/Fan_event.npy"
export G0="$DATA/corrected_g0.ply"
export FRAMES=100 WIDTH=640 HEIGHT=640

export INIT="$MERCY_ROOT/outputs/fan-init"
export CAMERA="$MERCY_ROOT/outputs/fan-inspection/inspection.json"
export BINDING="$INIT/initial-binding.npz"
export RUN="$MERCY_ROOT/outputs/Fan/feast_motion_discovery"

export WAFT_REPO=/root/autodl-tmp/WAFT
export WAFT_CFG=/path/to/matching-waft-config.json
export WAFT_CKPT=/root/autodl-tmp/WAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth
export COTRACKER_CKPT=/root/autodl-tmp/co-tracker/checkpoints/scaled_offline.pth
mkdir -p "$INIT" "$RUN"
```



****视频一致性：**** 本例将 `Fan.mkv` 作为统一 RGB 输入。WAFT 必须对同一份视频生成光流；如果旧的 `waft_flow.npy` 来自 `Fan10001-1000.mkv`，应重新生成，不能仅凭帧数相同而混用。WAFT 配置 `WAFT_CFG` 必须匹配权重。



## 2. 初始 Gaussian–Part Binding
****功能：**** 为 Canonical G0 中每个 Gaussian 确定刚体 Part ID，形成后续部件级运动估计的结构先验。若已有与 G0 精确对齐的 SAGA Binding，可直接跳至第 3 节。



### 2.1 Blender 表面部件导出（可选入口）
****通用命令****



```bash
blender --background --disable-autoexec "$DATA/$MODEL.blend" \
  --python "$MERCY_ROOT/scripts/motion/export_blender_parts.py" -- \
  --output "$INIT/blender-surface-parts.npz" \
  --frames "$FRAMES" --start 1 --step 10
```



****Fan 示例命令****



```bash
blender --background --disable-autoexec \
  /root/autodl-tmp/Mono_Data/Fan/Fan.blend \
  --python "$MERCY_ROOT/scripts/motion/export_blender_parts.py" -- \
  --output ../outputs/fan-init/blender-surface-parts.npz \
  --frames 100 --start 1 --step 10
```



输出 `blender-surface-parts.npz`：`rest_points`、`point_parts`、`object_matrices`、`metadata_json` 等。****FEAST 只使用静态表面与部件归属建立 Binding，不将 `object_matrices` 用作运动监督。****



### 2.2 从 fixed_camera.json 生成 Inspection
****功能：**** 将 Stage 1 固定相机转换为 MERCY Binding 所读取的 `camera.pose` 结构，不重新估计相机。



****通用命令****



```bash
mkdir -p "$(dirname "$CAMERA")"
python - <<'PY'
import os, json, math
import numpy as np
from pathlib import Path
src = json.loads((Path(os.environ['DATA']) / 'fixed_camera.json').read_text())
w2c = np.eye(4)
w2c[:3, :3] = np.asarray(src['R']).T
w2c[:3, 3] = src['T']
w2c[1:3] *= -1
pose = np.linalg.inv(w2c)
fovy = 2 * math.atan(src['height'] / (2 * src['fy']))
report = {'camera': {
    'pose': pose.tolist(), 'width': src['width'], 'height': src['height'],
    'fovy_degrees': math.degrees(fovy),
    'target': (pose[:3, 3] - src['distance'] * pose[:3, 2]).tolist()
}, 'models': []}
Path(os.environ['CAMERA']).write_text(json.dumps(report, indent=2) + '\n')
print(os.environ['CAMERA'])
PY
```



****Fan 示例命令****



```bash
# 执行上面的通用相机转换命令，使用第 1.3 节 Fan 环境变量。
# 输出：../outputs/fan-inspection/inspection.json
```



### 2.3 Blender 表面到 Gaussian 的几何绑定
****功能：**** 使用相机初始化和几何配准，将 Blender Part 标签近似映射到 `corrected_g0.ply` 的 Gaussian。



****通用命令****



```bash
python "$MERCY_ROOT/scripts/motion/bind_lego_parts.py" \
  --data "$DATA" \
  --parts "$INIT/blender-surface-parts.npz" \
  --inspection "$CAMERA" \
  --source-ply "$G0" \
  --output "$BINDING"
```



****Fan 示例命令****



```bash
python "$MERCY_ROOT/scripts/motion/bind_lego_parts.py" \
  --data /root/autodl-tmp/Mono_Data/Fan \
  --parts ../outputs/fan-init/blender-surface-parts.npz \
  --inspection ../outputs/fan-inspection/inspection.json \
  --source-ply /root/autodl-tmp/Mono_Data/Fan/corrected_g0.ply \
  --output ../outputs/fan-init/initial-binding.npz
```



输出 `initial-binding.npz`，包含 `labels: (N,)`、`registration: (4,4)`、`dynamic_indices` 等。Fan 本次绑定得到 ****N = 722,114、P = 12****；初始 `dynamic_indices = 286,271` 来自 Blender 判定，不代表 FEAST 最终动态数量。



## 3. WAFT：稠密光流
****功能：**** 由连续 RGB 帧估计像素级运动场，为动态部件发现提供观测。



****通用命令****



```bash
python -m FEAST.generate_waft_flow \
  --repo "$WAFT_REPO" --cfg "$WAFT_CFG" --checkpoint "$WAFT_CKPT" \
  --video "$VIDEO" --frames "$FRAMES" \
  --width "$WIDTH" --height "$HEIGHT" \
  --output "$RUN/waft_flow.npy"
```



****Fan 示例命令****



```bash
python -m FEAST.generate_waft_flow \
  --repo /root/autodl-tmp/WAFT \
  --cfg "$WAFT_CFG" \
  --checkpoint /root/autodl-tmp/WAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth \
  --video /root/autodl-tmp/Mono_Data/Fan/Fan.mkv \
  --frames 100 --width 640 --height 640 \
  --output ../outputs/Fan/feast_motion_discovery/waft_flow.npy
```



输出 `waft_flow.npy: (T−1, H, W, 2)`；Fan 对应 `(99, 640, 640, 2)`。



## 4. Stage A：2D Motion Discovery
****功能：**** 从连续光流的运动证据中选择候选 Gaussian 与可能运动的 Part，为后续轨迹验证提供高召回候选集合。



****通用命令****



```bash
python -m FEAST.flow_first_gaussians \
  --g0 "$G0" --binding "$BINDING" --camera "$CAMERA" \
  --flow-npy "$RUN/waft_flow.npy" \
  --min-part-fraction 0.40 \
  --output "$RUN/flow_first_trajectory_40"
```



****Fan 示例命令****



```bash
python -m FEAST.flow_first_gaussians \
  --g0 /root/autodl-tmp/Mono_Data/Fan/corrected_g0.ply \
  --binding ../outputs/fan-init/initial-binding.npz \
  --camera ../outputs/fan-inspection/inspection.json \
  --flow-npy ../outputs/Fan/feast_motion_discovery/waft_flow.npy \
  --min-part-fraction 0.40 \
  --output ../outputs/Fan/feast_motion_discovery/flow_first_trajectory_40
```



输出 `flow_candidate_parts.json`、`flow_candidates.npz`。Fan 本次候选为 `[0, 1, 2, 3, 4, 5]`，候选 Gaussian 数为 `280,872`；****候选不等于最终已验证动态 Part****。



## 5. Stage B：CoTracker3 + Event 运动验证
****功能：**** 联合长时序点轨迹、光流运动一致性和事件轮廓证据，确认候选 Part 的持续运动；保留证据不足的 Part 为 `unknown`，避免将事件响应直接等同于真实运动。



当前脚本要求 `--pose-init` 包含 `delta_g0: (T,P,4,4)`。本次采用恒等变换满足现有接口，**不使用 Blender 动画作为运动真值**；此文件仅为验证步骤的兼容输入，不代表后续要执行位姿估计。



****通用命令****



```bash
python - <<'PY'
import os
from pathlib import Path
import numpy as np
with np.load(os.environ['BINDING'], allow_pickle=False) as z:
    parts = int(np.max(z['labels'])) + 1
frames = int(os.environ['FRAMES'])
output = Path(os.environ['RUN']) / 'identity_part_poses.npz'
delta = np.broadcast_to(np.eye(4, dtype=np.float32), (frames, parts, 4, 4)).copy()
np.savez_compressed(output, delta_g0=delta)
print(output)
PY

python -m FEAST.event_track_confirm \
  --g0 "$G0" --binding "$BINDING" --camera "$CAMERA" \
  --video "$VIDEO" --events "$EVENTS" \
  --flow-npy "$RUN/waft_flow.npy" \
  --flow-candidates "$RUN/flow_first_trajectory_40/flow_candidate_parts.json" \
  --cotracker-checkpoint "$COTRACKER_CKPT" \
  --pose-init "$RUN/identity_part_poses.npz" \
  --frames "$FRAMES" --width "$WIDTH" --height "$HEIGHT" \
  --output "$RUN/confirmation_eventvis_tracks"
```



****Fan 示例命令****



```bash
python -m FEAST.event_track_confirm \
  --g0 /root/autodl-tmp/Mono_Data/Fan/corrected_g0.ply \
  --binding ../outputs/fan-init/initial-binding.npz \
  --camera ../outputs/fan-inspection/inspection.json \
  --video /root/autodl-tmp/Mono_Data/Fan/Fan.mkv \
  --events /root/autodl-tmp/Mono_Data/Fan/Fan_event.npy \
  --flow-npy ../outputs/Fan/feast_motion_discovery/waft_flow.npy \
  --flow-candidates ../outputs/Fan/feast_motion_discovery/flow_first_trajectory_40/flow_candidate_parts.json \
  --cotracker-checkpoint /root/autodl-tmp/co-tracker/checkpoints/scaled_offline.pth \
  --pose-init ../outputs/Fan/feast_motion_discovery/identity_part_poses.npz \
  --frames 100 --width 640 --height 640 \
  --output ../outputs/Fan/feast_motion_discovery/confirmation_eventvis_tracks
```



输出 `motion_confirmation.json`（`moving_parts`）、`tracking_evidence.npz`、`cotracker_raw_tracks.mp4`、`event_part_debug/`。Fan 初始事件对应 Part 为 `[1, 2, 5]`，属于事件佐证，并非最终 `moving_parts`。



## 6. Stage C：Dynamic Gaussian Binding
****功能：**** 将 Stage A 的 Gaussian 运动候选与 Stage B 确认的运动 Part 结合，形成动态 Gaussian 索引集合，同时保留原始 Part 归属。



****通用命令****



```bash
python -m FEAST.finalize_motion_binding \
  --binding "$BINDING" \
  --flow-candidates "$RUN/flow_first_trajectory_40/flow_candidates.npz" \
  --confirmation "$RUN/confirmation_eventvis_tracks/motion_confirmation.json" \
  --output "$RUN/final_binding"
```



****Fan 示例命令****



```bash
python -m FEAST.finalize_motion_binding \
  --binding ../outputs/fan-init/initial-binding.npz \
  --flow-candidates ../outputs/Fan/feast_motion_discovery/flow_first_trajectory_40/flow_candidates.npz \
  --confirmation ../outputs/Fan/feast_motion_discovery/confirmation_eventvis_tracks/motion_confirmation.json \
  --output ../outputs/Fan/feast_motion_discovery/final_binding
```



输出 `final_binding.npz`。




## 7. 最终输出（FEAST 截止位置）

```text
outputs/<MODEL>/feast_motion_discovery/
├── waft_flow.npy
├── flow_first_trajectory_40/
│   ├── flow_candidate_parts.json
│   └── flow_candidates.npz
├── confirmation_eventvis_tracks/
│   ├── motion_confirmation.json
│   ├── tracking_evidence.npz
│   └── event_part_debug/
└── final_binding.npz
```

最终以 **`final_binding.npz`** 作为 FEAST V2 输出。本文不包含 PnP、SE(3) 优化或 Gaussian Motion Propagation，也不包含这些步骤的命令行。