# 新增QA任务生成器使用说明

本文档介绍新增的两个QA生成脚本：度量测量 (Metric Measurement) 和空间关系 (Spatial Relations)。

## 目录
- [前置条件](#前置条件)
- [度量测量QA](#度量测量qa)
- [空间关系QA](#空间关系qa)
- [完整使用流程](#完整使用流程)
- [配置选项](#配置选项)
- [输出格式](#输出格式)

## 前置条件

### 1. 修改后的 executable_plan.py
新任务需要 `executable_plan.py` 保存完整的物体信息。已对脚本进行以下修改：

**新增函数**:
```python
def _extract_visible_objects_details(event):
    """提取可见物体的详细信息,包括位置和bounding box"""
```

**保存的新字段**:
- `visible_objects`: 每个可见物体的详细信息
  - `objectId`, `objectType`
  - `position`: {x, y, z}
  - `rotation`: {x, y, z}
  - `axisAlignedBoundingBox`: {center, size}
  - `distance`: 到相机的距离
- `camera_position`: {x, y, z}
- `camera_rotation`: {x, y, z}

### 2. 运行环境
- Linux系统 (AI2-THOR要求)
- Python 3.10+
- AI2-THOR >= 5.0

### 3. 数据准备
首先运行 `executable_plan.py` 生成episode数据：
```bash
python executable_plan.py
```

输出目录结构：
```
outputs/video/train_15_TissueBox_20251210_125749/
  └── agent_1_*/
      ├── img_00000.png
      ├── img_00000.json   # 包含新增的详细信息
      ├── img_00001.png
      ├── img_00001.json
      └── ...
```

---

## 度量测量QA

### 功能
生成三类度量测量问答：
1. **相机到物体距离** - 测量相机与场景中物体的精确距离
2. **物体间距离** - 测量两个物体之间的距离
3. **物体尺寸** - 测量物体的宽度/高度/深度/体积

### 基础用法
```bash
python generate_metric_qas.py <episode_dir> \
  --output-dir outputs/metric \
  --output-name metric_questions.json
```

### 完整参数
```bash
python generate_metric_qas.py \
  /data5/zhuangyunhao/outputs/video/train_15_TissueBox_20251210_125749 \
  --agent-prefix agent_1 \
  --camera-distance-questions 10 \
  --object-distance-questions 8 \
  --size-questions 12 \
  --seed 2025 \
  --output-dir outputs/metric \
  --output-name metric_questions.json
```

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `episode_dir` | (必需) | Episode数据目录 |
| `--agent-prefix` | `agent_1` | 智能体目录前缀 |
| `--camera-distance-questions` | 10 | 相机距离问题数 |
| `--object-distance-questions` | 8 | 物体间距离问题数 |
| `--size-questions` | 12 | 尺寸问题数 |
| `--seed` | 2025 | 随机种子 |
| `--output-dir` | `outputs/metric` | 输出目录 |
| `--output-name` | `metric_questions.json` | 输出文件名 |

### 单位配置
可在脚本顶部统一修改：

```python
# generate_metric_qas.py 第10-12行
DISTANCE_UNIT = "cm"  # 可改为 "m" 或 "mm"
SIZE_UNIT = "cm"
DECIMAL_PLACES = 2    # 保留小数位数
```

### 问题示例

**类型1: 相机到物体距离**
```json
{
  "question": "Calculate the distance from the camera to the Apple.",
  "options": [
    "145.23 cm",
    "210.45 cm",
    "98.76 cm",
    "182.34 cm"
  ],
  "answer": "145.23 cm",
  "metadata": {
    "type": "camera_to_object_distance",
    "object": "Apple",
    "distance_value": 145.23,
    "unit": "cm"
  }
}
```

**类型2: 物体间距离**
```json
{
  "question": "What is the distance between the Apple and the Fridge?",
  "options": [
    "234.56 cm",
    "187.32 cm",
    "312.45 cm",
    "156.78 cm"
  ],
  "answer": "234.56 cm"
}
```

**类型3: 物体尺寸**
```json
{
  "question": "What is the height of the Chair?",
  "options": [
    "95.40 cm",
    "120.50 cm",
    "78.30 cm",
    "145.60 cm"
  ],
  "answer": "95.40 cm"
}
```

**类型4: 物体体积**
```json
{
  "question": "What is the approximate volume of the Apple?",
  "options": [
    "125000.00 cm³",
    "80000.00 cm³",
    "200000.00 cm³",
    "160000.00 cm³"
  ],
  "answer": "125000.00 cm³"
}
```

### 干扰选项生成规则
- 相对误差 **> 20%**
- 随机在正确值的 0.75x ~ 1.8x 范围内生成
- 确保所有选项都为正值

---

## 空间关系QA

### 功能
生成三类空间关系问答：
1. **第一人称方向** - 物体相对于相机的方向 (8方向)
2. **距离比较** - 比较两个物体谁离相机更远/更近
3. **大小比较** - 比较两个物体的体积大小

### 基础用法
```bash
python generate_spatial_qas.py <episode_dir> \
  --output-dir outputs/spatial \
  --output-name spatial_questions.json
```

### 完整参数
```bash
python generate_spatial_qas.py \
  /data5/zhuangyunhao/outputs/video/train_15_TissueBox_20251210_125749 \
  --agent-prefix agent_1 \
  --egocentric-questions 12 \
  --distance-comparison-questions 10 \
  --size-comparison-questions 10 \
  --seed 2025 \
  --output-dir outputs/spatial \
  --output-name spatial_questions.json
```

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `episode_dir` | (必需) | Episode数据目录 |
| `--agent-prefix` | `agent_1` | 智能体目录前缀 |
| `--egocentric-questions` | 12 | 第一人称方向问题数 |
| `--distance-comparison-questions` | 10 | 距离比较问题数 |
| `--size-comparison-questions` | 10 | 大小比较问题数 |
| `--seed` | 2025 | 随机种子 |

### 8方向定义

方向基于相机的朝向 (yaw angle):

| 方向 | 角度范围 | 描述 |
|------|---------|------|
| **front** | -30° ~ 30° | 正前方 |
| **front-right** | 15° ~ 75° | 右前方 |
| **right** | 45° ~ 135° | 正右方 |
| **back-right** | 105° ~ 165° | 右后方 |
| **back** | 135° ~ -135° | 正后方 |
| **back-left** | -165° ~ -105° | 左后方 |
| **left** | -135° ~ -45° | 正左方 |
| **front-left** | -75° ~ -15° | 左前方 |

**歧义处理**: 当物体在边界角度时 (如30°,可以是"front"或"front-right"),脚本会：
- 优先选择4方向 (front/right/back/left)
- 避免在干扰选项中同时出现多个正确答案

### 问题示例

**类型1: 第一人称方向**
```json
{
  "question": "From the camera's perspective, in which direction is the Apple?",
  "options": [
    "front-right",
    "back",
    "left",
    "front-left"
  ],
  "answer": "front-right",
  "metadata": {
    "type": "egocentric_direction",
    "object": "Apple",
    "direction": "front-right",
    "angle": 45.23,
    "all_valid_directions": ["front-right", "right"]
  }
}
```

**类型2: 距离比较**
```json
{
  "question": "Which object is farther from the camera: the Apple or the Bowl?",
  "options": [
    "Apple",
    "Bowl"
  ],
  "answer": "Bowl",
  "metadata": {
    "type": "distance_comparison_farther",
    "object1": "Apple",
    "object2": "Bowl",
    "distance1": 1.4523,
    "distance2": 2.3456
  }
}
```

**类型3: 大小比较**
```json
{
  "question": "Which object is larger: the Chair or the Table?",
  "options": [
    "Table",
    "Chair"
  ],
  "answer": "Table",
  "metadata": {
    "type": "size_comparison",
    "object1": "Chair",
    "object2": "Table",
    "volume1": 0.072,
    "volume2": 0.456
  }
}
```

---

## 完整使用流程

### Step 1: 生成Episode数据
```bash
# 确保在Linux环境
export PROCTHOR_SPLIT=train
python executable_plan.py
```

### Step 2: 生成度量测量QA
```bash
python generate_metric_qas.py \
  /data5/zhuangyunhao/outputs/video/train_15_TissueBox_20251210_125749 \
  --camera-distance-questions 20 \
  --object-distance-questions 15 \
  --size-questions 20 \
  --output-dir outputs/metric
```

### Step 3: 生成空间关系QA
```bash
python generate_spatial_qas.py \
  /data5/zhuangyunhao/outputs/video/train_15_TissueBox_20251210_125749 \
  --egocentric-questions 20 \
  --distance-comparison-questions 15 \
  --size-comparison-questions 15 \
  --output-dir outputs/spatial
```

### Step 4: 批量处理多个Episode
```bash
# 为所有episode生成度量QA
for episode_dir in /data5/zhuangyunhao/outputs/video/*/; do
  episode_name=$(basename "$episode_dir")
  python generate_metric_qas.py \
    "$episode_dir" \
    --output-dir "outputs/metric/$episode_name" \
    --output-name "metric_qas.json"
done

# 为所有episode生成空间QA
for episode_dir in /data5/zhuangyunhao/outputs/video/*/; do
  episode_name=$(basename "$episode_dir")
  python generate_spatial_qas.py \
    "$episode_dir" \
    --output-dir "outputs/spatial/$episode_name" \
    --output-name "spatial_qas.json"
done
```

---

## 配置选项

### 修改度量单位
编辑 `generate_metric_qas.py`:
```python
# 第10-12行
DISTANCE_UNIT = "m"     # 改为米
DECIMAL_PLACES = 3      # 保留3位小数
```

### 修改方向定义
编辑 `generate_spatial_qas.py`:
```python
# 第20-29行
DIRECTION_DEFINITIONS = {
    "front": (-20, 20),  # 缩小前方范围到±20度
    # ...
}
```

### 调整干扰项误差范围
编辑 `generate_metric_qas.py` 中的 `generate_*_question` 函数:
```python
# 例如第239行
error_factor = random.uniform(0.30, 1.0)  # 误差范围30%-100%
```

---

## 输出格式

### 度量测量输出
```json
{
  "metadata": {
    "episode_dir": "/path/to/episode",
    "total_questions": 30,
    "camera_distance_questions": 10,
    "object_distance_questions": 8,
    "size_questions": 12,
    "distance_unit": "cm",
    "size_unit": "cm",
    "decimal_places": 2
  },
  "questions": [
    { "question": "...", "options": [...], "answer": "...", "metadata": {...} },
    ...
  ]
}
```

### 空间关系输出
```json
{
  "metadata": {
    "episode_dir": "/path/to/episode",
    "total_questions": 32,
    "egocentric_questions": 12,
    "distance_comparison_questions": 10,
    "size_comparison_questions": 10
  },
  "questions": [
    { "question": "...", "options": [...], "answer": "...", "metadata": {...} },
    ...
  ]
}
```

---

## 故障排查

### 问题: 生成的问题数少于预期
**原因**: 帧数据中可见物体数量不足

**解决方案**:
- 增加 `max_attempts` (默认是帧数×10)
- 运行 `executable_plan.py` 生成更多帧
- 确保物体在视野内且未被遮挡

### 问题: "No frames found"
**原因**: Episode目录路径错误或agent前缀不匹配

**解决方案**:
```bash
# 检查目录结构
ls /data5/zhuangyunhao/outputs/video/train_15_TissueBox_20251210_125749/

# 查看agent子目录名称
ls /data5/zhuangyunhao/outputs/video/train_15_TissueBox_20251210_125749/

# 使用正确的前缀
python generate_metric_qas.py <episode_dir> --agent-prefix agent_1_train_15_TissueBox_20251210_125749
```

### 问题: Metadata缺少字段
**原因**: 使用了旧版本的 `executable_plan.py`

**解决方案**:
- 确保使用修改后的 `executable_plan.py`
- 重新运行 `executable_plan.py` 生成新数据
- 检查frame JSON是否包含 `visible_objects`, `camera_position`, `camera_rotation`

---

## 技术细节

### 距离计算
使用3D欧氏距离:
```python
distance = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
```

### 体积计算
使用轴对齐边界框:
```python
volume = width × height × depth
```

### 方向计算
1. 计算物体相对相机的绝对角度
2. 减去相机yaw角度得到相对角度
3. 归一化到 [-180°, 180°]
4. 匹配对应的方向区间

---

## 许可与贡献
这些脚本遵循项目主许可证。欢迎提交issue和PR。
