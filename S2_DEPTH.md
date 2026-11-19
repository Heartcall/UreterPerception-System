# S2 单帧深度实验 · 2026-11-19

**首轮采用 Depth Anything V2 Small 作为机器人时序纠错的单帧前端。** 本次比较针对本机、固定分辨率和两个 Small 检查点。两者均只看当前 RGB。DA3 的多视图和位姿条件能力留作后续时序对照。

## 数据和冻结模型对照

直接对本地 KidneyDepthMetric.zip 的成员逐一配对，得到 21,697 对 RGB 和浮点 TIFF。以轨迹编号分组，材质、照明、标准轨迹和同编号的随机扰动版本均在同一侧。编号 1、2 用于训练（63 个序列，13,795 帧），3、4 用于验证（24 个序列，4,702 帧），5、6 用于测试（16 个序列，3,200 帧）。验证和测试都含 ct 与 stones 场景。每个测试编号仅代表一种基础路径，多个渲染版本共享几何，结论限于本次模拟域。[划分清单](runs/s2/trajectory_split.json)

冻结对照在每个验证序列均匀取 8 帧，共 192 帧。每帧用相同随机种子抽 8,192 对有效像素，剔除参考深度相等的像素对。远近判断正确计 1 分，预测并列计半分，无效预测计 0 分。V2 的相对逆深度与 DA3 的相对深度分别按各自方向判断。这里的排序以原始 TIFF 数值为参考，避免假设其单位或用测试真值对齐。

| 冻结模型 | 远近排序正确率 | ct 场景 | stones 场景 | 单帧 P50 / P95 |
|---|---:|---:|---:|---:|
| Depth Anything V2 Small | 80.45% | 78.99% | 81.19% | 28.1 / 37.1 ms |
| DA3-Small | 67.84% | 63.97% | 69.77% | 42.0 / 54.5 ms |

V2 在本次 24 个验证序列上均取得更高的排序正确率。stones 一列统计整幅含石场景，结石像素的单独误差仍需同帧掩膜。延迟在 RTX 4070 Ti 上串行测量，batch=1，预热后 GPU 同步计时，包含 RGB 缩放到 518 × 518、归一化、网络前向、回到原图网格与 CPU。两版使用同一输入预处理、FP32 张量和 PyTorch 默认卷积设置。压缩包读取、分割、时序融合和显示不在计时范围内。[V2 逐帧记录](runs/s2/benchmark/v2-small/frames.jsonl) · [DA3 逐帧记录](runs/s2/benchmark/v3-small/frames.jsonl)

## KidneyDepth 监督微调

先从每条训练序列取一帧，以统一的对数线性映射把冻结输出校准到原始 TIFF 深度。V2 的映射输入为 −log(1 + 相对逆深度)，DA3 使用 log(相对深度)。映射在训练集拟合一次并固定。GT 保留有限正值，不在测试时做逐帧尺度拟合。

两版均冻结骨干、更新解码器。V2 更新 neck 和 head，共 2,728,513 个参数；DA3 更新其 head，共 3,873,854 个参数。两轮各从每条训练序列随机抽 4 帧，共 504 次参数更新，随机种子 42，AdamW 学习率 0.0001，使用对数深度的 Smooth L1 损失。初始冻结权重也参与验证集选优，两版均选中第 2 轮。相同的抽样和更新次数使两种结构可对照，尚未考察充分训练后的性能上限。

| 模型 | 调整尺度后冻结模型的验证 AbsRel | 微调后验证 AbsRel |
|---|---:|---:|
| V2 Small | 0.7814 | 0.4366 |
| DA3-Small | 0.9505 | 0.4374 |

验证使用每条验证序列的 4 帧，共 96 帧。AbsRel 为每个有效像素的绝对误差除以参考深度，先按帧、再按序列平均，越低越好。误差在近处会获得较高权重。两个微调模型的验证误差非常接近。先导实验还单独尝试过仅更新 V2 最后输出头，此时验证 AbsRel 为 0.6900。正式对照使用包含多尺度特征融合层的完整深度解码器。[V2 微调清单](runs/s2/finetuned/v2-decoder/manifest.json) · [DA3 微调清单](runs/s2/finetuned/v3-small/manifest.json)

### 全部留出轨迹

以下结果覆盖 16 个序列各自完整的 200 帧。两种模型的 3,200 个帧键逐项相同，已从逐帧结果独立复算汇总。

| 模型 | 冻结＋训练集尺度校准 AbsRel | 微调后 AbsRel | 微调后 ct | 微调后 stones |
|---|---:|---:|---:|---:|
| V2 Small | 0.5438 | **0.3375** | 0.4116 | 0.2635 |
| DA3-Small | 0.6464 | **0.4102** | 0.4836 | 0.3368 |

两者均从 KidneyDepth 监督中获益。完整测试上，V2 的平均相对误差低 0.0726，且初筛延迟较低，故采用 V2 进入时序纠错实验。测试涵盖两个留出的基础轨迹编号，无法据此评价跨设备或跨患者泛化。0.3375 是对 TIFF 参考值的平均绝对相对误差；石面毫米误差及光纤尖端误差需另测。[V2 全量逐帧记录](runs/s2/test-full/v2/frames.jsonl) · [DA3 全量逐帧记录](runs/s2/test-full/v3/frames.jsonl)

## 与 MICCAI 2025 KidneyDepth 论文的同基准比较

**当前 V2 Small 的单帧结果在两项共同指标上均落后于论文报告的 V2 和 ZoeDepth。** 论文表 1 的定量测试使用 KidneyDepthSLAM；前一节的 0.3375 来自 KidneyDepthMetric 留出轨迹。为比较同一数据来源，本次固定训练好的权重与尺度映射，在本地 KidneyDepthSLAM 五个序列上分别取第 0、10、…、2290 帧，共 1,150 对 RGB 和深度图。逐图在原始 1042 × 1080 网格的有限正真值像素上评分，再按图平均。每张图均有 1,125,360 个有效像素。论文未列出逐帧评测索引，本次比较仍存在抽样和具体实现差异。[原论文 §3.2、表 1](https://papers.miccai.org/miccai-2025/paper/2047_paper.pdf) · [发布数据](https://zenodo.org/records/17294033)

| 方法 | AbsRel ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | RMSE-log ↓ |
|---|---:|---:|---:|---:|---:|
| 论文 Depth Anything V2，Complete | 0.139 | 86.9% | 未报告 | 未报告 | 未报告 |
| 论文 ZoeDepth，Complete | 0.179 | 77.3% | 未报告 | 未报告 | 未报告 |
| 本项目 V2 Small | 0.3861 | 39.23% | 68.68% | 86.35% | 0.4084 |
| 本项目 DA3-Small | 0.4488 | 38.17% | 69.64% | 87.53% | 0.4132 |

AbsRel 为平均绝对误差除以对应像素真值后取平均，即 mean(|预测−真值| / 真值)。δk 为满足 max(预测/真值，真值/预测) < 1.25ᵏ 的像素比例。RMSE-log 使用自然对数。按 TIFF 原生单位计算，本项目 V2 的逐图 MAE / RMSE 均值为 2.566 / 3.093，DA3 为 2.693 / 3.302；这两项不作为真实软镜的毫米误差。论文表 1 只提供 AbsRel 和 δ1。[本地汇总](runs/s2/paper-comparison/manifest.json) · [逐帧记录](runs/s2/paper-comparison/frames.jsonl)

| 序列 | V2 AbsRel / δ1 | DA3 AbsRel / δ1 |
|---|---:|---:|
| Seq_01 | 0.3607 / 42.71% | 0.4182 / 39.01% |
| Seq_02 | 0.3796 / 42.15% | 0.4667 / 39.60% |
| Seq_03 | 0.3720 / 36.37% | 0.4165 / 37.07% |
| Seq_04 | 0.3705 / 40.63% | 0.4339 / 39.99% |
| Seq_05 | 0.4479 / 34.27% | 0.5085 / 35.20% |

五个序列的 AbsRel 均由本地 V2 取得较低值，两个模型在 Seq_05 上的 AbsRel 均最高。相对论文 V2，本地 V2 的 AbsRel 高 0.2471，δ1 低 47.67 个百分点。当前两轮抽样微调共 504 次更新，冻结骨干；作者报告 40 轮训练，并使用亮度、噪声等增强。训练覆盖、模型配置和尺度处理都可能影响差距，当前对照无法把差距归因给某个因素。论文对真实 fURS 的比较采用视觉检查，因此此表能评价合成肾腔的跨材质泛化，临床石面和光纤间距的精度仍需独立真值。[原论文 §3.2](https://papers.miccai.org/miccai-2025/paper/2047_paper.pdf)

复现时在下方所列环境运行，输出指定为空目录。

```powershell
python tools/compare_kidneydepth_slam.py --archive ../KidneyDepth/KidneyDepthSLAM.zip --assets runs/s2 --output runs/repeat/paper-comparison --stride 10
```

## 真实模拟体回放与时序纠错

已用正式 V2 微调权重回放本地 video_1 和 video_2 的前 12 帧，共 24 张 400 × 400 深度图。逐帧验证了深度形状、数值有效性以及三维距离字段的缺失状态。预测图保存在每段回放的 depth 目录。色条为共用的训练数据原生尺度；在真实相机的水中标定、尺度和参考测量齐备前，系统将这些图作为相对深度使用。[两段原图与深度对照](runs/s2/replay-decoder/preview.png)

下一步固定同一前端，在带时间戳的短窗口内估计当前石面与镜头的相对运动。镜体递送、周转、弯转给出相机运动初值和误差范围，用过去帧与当前帧中稳定对应的石面点修正位姿。把历史深度投影到当前视角，做可见性检验后，以视差和重投影残差决定是否以及多大程度修正当前深度。纯旋转只能检验方向一致性；缺少可靠平移基线或独立尺度参考时应保留尺度不确定度。光纤的独立运动及反光高亮区域须从静态石面对应中剔除。固定伸出长度用于约束尖端几何。

同一时间窗和截止时刻下比较单帧、仅历史视觉、仅机器人传播、两者联合，以及 DA3 原生的因果多帧模式。使用同一批测试轨迹，分别报告石面深度误差、沿光纤轴距离误差、落点误差、可用率及新增延迟。当前两段模拟体视频缺少同步机器人反馈和独立三维真值，现阶段能验证深度图生成与回放，机器人纠错效果仍待实验。[实施计划](PLAN.md)

首个纠错实验可将状态收敛到两个量：机器人名义相机位姿增量的小幅修正，以及局部石面深度的尺度修正。对于历史帧石面点，用历史深度和水中相机内参反投影，经修正后的位姿变换到当前帧；其像素位置应接近实际跟踪点，其光轴深度应与修正后的当前预测一致。用运动学及单帧深度作先验，按对应质量加权这两种残差。须同时检查视差和位姿先验的平移尺度；若平移量与深度尺度可以相互抵消，保留原尺度并标明不确定度。用有偏动作输入检验视觉能否纠正运动先验，再用短遮挡检验动作能否补偿视觉缺失。

## 复现

从 UreterPerception 目录运行。项目运行目录中保留了两份基础权重、微调权重、独立依赖和 DA3 官方源码。DA3 源码固定于 Git 提交 `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`，按 Git 文件摘要校验。两份 safetensors 基础权重按仓库 LFS 摘要校验，完整 SHA-256 见训练清单。当前环境为 uretersam、PyTorch 2.11.0+cu128。复现实验时使用新的输出目录。

```powershell
conda activate uretersam
$env:PYTHONPATH=(Resolve-Path runs/s2/deps).Path+';'+(Resolve-Path runs/s2/da3-src/src).Path
$env:OMP_NUM_THREADS='1'
$archive='../KidneyDepth/KidneyDepthMetric.zip'
$weights='runs/s2/checkpoints/Depth-Anything-V2-Small-hf'
python run.py eval depth-split --archive $archive
python run.py eval depth-benchmark --model v2-small --archive $archive --weights $weights --output runs/repeat/frozen-v2
python run.py run finetune-depth --model v2-small --archive $archive --weights $weights --output runs/repeat/tuned-v2
python run.py eval adapted-depth --archive $archive --weights $weights --depth-run runs/repeat/tuned-v2 --per-sequence 200 --output runs/repeat/test-v2
python run.py run replay --video data/videos/video_1.mp4 --g0-cache ../UreterSAM/outputs/v7_v8_video_1 --depth-weights $weights --depth-run runs/repeat/tuned-v2 --limit 12 --output runs/repeat/video_1
```

DA3 的模型参数用 `v3-small`，基础权重目录用 `runs/s2/checkpoints/DA3-SMALL`，相应输出目录单独命名。[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) · [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) · [KidneyDepth 原论文](https://papers.miccai.org/miccai-2025/0469-Paper2047.html)
