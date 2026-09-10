# 数据集说明

本目录保存 CRF-SAM3 使用的数据路径与说明；算法实现位于 `../Sa2VA-main/`。项目涉及 **MeViS、DAVIS17、Ref-Youtube-VOS、ReVOS 四个数据集**。

## 目录与用途

| 目录（相对 `data/video_datas/`） | 项目用途 | 划分 |
|---|---|---|
| `mevis/` | 三数据集联合训练、MeViS 评测 | `train`、`valid_u`、`valid` |
| `davis17/` | 三数据集联合训练、DAVIS17 评测 | `train`、`valid` |
| `Ref-Youtube-VOS/` | 联合训练、类别监督和 Ref-Youtube-VOS 评测 | `train`、`valid` |
| `revos/` | ReVOS 原始标注、来源视频与最终评测 | 根目录 train/valid 元数据文件 |
| `revos_prepared/` | 将 ReVOS 整理成统一加载器使用的训练布局 | `train`、`valid` |

`revos_prepared` 与 `revos` 使用同一数据，不能作为独立数据集重复计数。脚本中的 `refytros` 名称指向本项目的 Ref-Youtube-VOS 数据路径。

## 元数据规模与本地媒体样本

下表的完整规模按当前元数据中的 `videos` 和 `expressions` 计数，并非声称磁盘上存在全部视频帧；样本数量按 [sample_inventory.json](../sample_inventory.json) 统计。它们描述本项目使用的标注版本，不代替官方数据集总规模。

| 数据布局 | 划分 | 元数据视频数 | 元数据表达式数 | 清单内采样视频数 | 清单内 RGB 帧数 |
|---|---|---:|---:|---:|---:|
| mevis | train | 1662 | 23051 | 8 | 64 |
| mevis | valid_u | 50 | 793 | 8 | 64 |
| mevis | valid | 140 | 2236 | 8 | 64 |
| davis17 | train | 60 | 572 | 8 | 64 |
| davis17 | valid | 30 | 244 | 8 | 64 |
| Ref-Youtube-VOS | train | 3471 | 12913 | 8 | 64 |
| Ref-Youtube-VOS | valid | 202 | 834 | 8 | 64 |
| revos_prepared | train | 626 | 29252 | 13 | 104 |
| revos_prepared | valid | 416 | 5822 | 15 | 120 |

清单共 84 条视频记录、672 帧 RGB。每条记录包含视频 ID、表达式、元数据来源、原视频帧数和实际抽样帧路径。ReVOS 样本按来源与表达式类型分层覆盖；这些样本用于理解数据结构，不能代替完整测试集，也不代表最终训练实际使用的样本数。类别监督划分及低召回过滤会进一步改变训练集合。

## 标注与文件含义

- `JPEGImages/<video_id>/`：视频帧；`Annotations/<video_id>/`：部分数据集提供的 PNG 对象掩码。
- `meta_expressions*.json`：视频、帧序列与指代表达关系；表达式中的 `exp` 为文本，`obj_id` 指向目标对象，`anno_id` 在相应布局中关联掩码标注。一个表达式可能对应多个对象。
- `mask_dict.json` / `mask_dict.pkl`：加载器使用的序列掩码字典；各数据集格式由对应加载器解释，不能直接互换。
- ReVOS 的 `type_id` 在评测代码中区分 `0`（referring）、`1`（reason）和 `2`（单独处理的类型）。最终指标应以评测文件及其协议为准。

## Git 与复现实验

Git 仅保存数据说明、目录占位和样本清单；原始帧、完整标注、掩码字典和权重不随仓库发布。本地媒体同样只有抽样内容，旧样本可能额外保留，所以上表不是目录中所有文件的盘点。

复现实验时，将完整数据恢复到 `data/video_datas/` 对应目录，配置数据根路径、模型路径与缓存路径。ReVOS 布局转换使用 [prepare_revos_mevis_layout.py](../Sa2VA-main/tools/prepare_revos_mevis_layout.py)。配置中的服务器绝对路径需要按环境调整。

实验配置与论文结果见 [论文材料索引](../PAPER_EVIDENCE_MAP.md)。
