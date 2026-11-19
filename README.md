# UreterPerception

这是「面向机器人辅助输尿管软镜的运动学约束时序三维空间感知」的本地研究代码。当前版本可回放 UreterSAM 的正式分割输出、比较和微调 Depth Anything V2 Small 与 DA3-Small、用合成真值验证三维几何，并检查机器人日志的时间对齐。实验结果见 [S0 分割与回放](S0_RESULTS.md)和 [S2 单帧深度](S2_DEPTH.md)。

## 目录与接口

| 位置 | 职责 |
|---|---|
| `src/ureter_perception/adapters/` | UreterSAM 缓存接入、KidneyDepth ZIP 解码、机器人数据源协议和 CSV 接入。 |
| `src/ureter_perception/segmentation.py`、`depth.py` | 后端调用协议、输出类型转换，以及像素域和数据有效性检查。 |
| `src/ureter_perception/math/` | 刚体变换、相机投影、拟合、平面最近点及射线求交。可以脱离相机和模型测试。 |
| `src/ureter_perception/datasets/` | 记录路径的组织、序列标识和样本键。 |
| `src/ureter_perception/io/` | 读视频、PNG、JSON，写逐帧记录和运行清单。 |
| `src/ureter_perception/geometry.py`、`targeting.py` | 从带度量深度的分割结果拟合光纤和局部矩形平面，计算平面范围、最短距离、沿轴距离和几何交点。 |
| `src/ureter_perception/pipeline.py` | 同一逐帧处理函数，供缓存回放和合成场景运行。 |
| `tests/fixtures/` | 仅含可检查的小型合成输入。 |

完整设计和后续接口见 [系统代码结构与接口设计](../系统代码结构与接口设计.md)。

## 本地运行

以下命令从本目录运行。已有的 `uretersam` 环境可执行视频回放和 NumPy 测试。读取 KidneyDepth 的浮点 TIFF 时使用包含 `tifffile` 的环境，本机已有 `FURS`。代码支持 Python 3.10 及以上版本。使用仓库入口 `run.py` 可直接复用环境。

```powershell
conda activate uretersam
python run.py run replay --video ../UreterSAM/data/videos/video_1.mp4 --g0-cache ../UreterSAM/outputs/v7_v8_video_1 --output runs/video1
python run.py run synthetic --input tests/fixtures/known_plane.json --output runs/known_plane
```

每次运行使用一个新的输出目录。成功生成 `frames.jsonl` 和 `manifest.json`。真实 UreterSAM 缓存回放会检查原视频和掩膜校验值，保留 G0 二维尖端估计及时间来源。当前这条真实回放没有标定后的度量深度输入，三维字段会给出缺失原因。合成场景使用已知度量深度，预期沿射线距离为 6 mm。

## 单模块运行和测试

```powershell
conda activate uretersam
python run.py eval segmentation --video ../UreterSAM/data/videos/video_1.mp4 --g0-cache ../UreterSAM/outputs/v7_v8_video_1 --limit 12
python run.py eval geometry --input tests/fixtures/known_plane.json
python run.py eval targeting --input tests/fixtures/known_ray.json
python run.py eval alignment --input tests/fixtures/robot.csv --clock-id fixture --capture-ns 100 --decision-ns 100 --max-age-ns 70

$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p test_math.py -v

conda activate FURS
python run.py eval depth --archive ../KidneyDepth/KidneyDepthMetric.zip --trajectory ct/flesh/dynamic/traj_1 --index 0
```

KidneyDepth 入口按样本直接读取 ZIP，保留数据集原始深度值。要作为度量观测使用，调用方须确认其尺度、相机 Z 深度或射线距离定义及像素域。机器人回放按事件时间和到达时间选择样本，要求输入已映射到同一时钟。

已接入微调后的单帧模型用于模拟体视频回放。平台实时采集、软镜运动学与时序纠错仍需实际设备接口、标定和同步数据。当前三维几何在合成场景验证，模拟体预测保留原生尺度和有效性标记，毫米落点精度待独立评价。
