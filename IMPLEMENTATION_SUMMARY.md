# 新增QA任务实现总结

## 概述
为项目新增了两类空间推理问答生成器:
1. **度量测量 (Metric Measurement, MM)** - `generate_metric_qas.py`
2. **空间关系 (Spatial Relations, SR)** - `generate_spatial_qas.py`

## 修改的文件

### 1. executable_plan.py (修改)
**修改内容**:
- 新增函数 `_extract_visible_objects_details()` (第89-112行)
  - 提取可见物体的详细信息 (位置、旋转、bounding box、距离)

- 修改帧保存逻辑 (第411-435行)
  - 保存 `visible_objects`: 物体详细信息列表
  - 保存 `camera_position`: 相机位置 {x, y, z}
  - 保存 `camera_rotation`: 相机旋转 {x, y, z}

**影响**:
- 向后兼容,不影响现有QA生成器
- 新增字段为度量和空间QA提供必要数据

### 2. generate_metric_qas.py (新增)
**功能**: 生成度量测量问答

**问题类型**:
1. 相机到物体距离 (10个)
2. 物体间距离 (8个)
3. 物体尺寸 - 宽/高/深/体积 (12个)

**核心函数**:
- `euclidean_distance_3d()`: 3D欧氏距离计算
- `calculate_bbox_dimensions()`: 从bbox计算尺寸
- `meters_to_unit()`: 单位转换 (可配置)
- `generate_camera_distance_question()`: 生成相机距离问题
- `generate_object_distance_question()`: 生成物体间距离问题
- `generate_size_question()`: 生成尺寸问题

**配置**:
```python
DISTANCE_UNIT = "cm"      # 可改为 "m" 或 "mm"
SIZE_UNIT = "cm"
DECIMAL_PLACES = 2         # 保留小数位数
```

**干扰项生成**:
- 相对误差 > 20%
- 在 0.75x ~ 1.8x 范围随机生成

### 3. generate_spatial_qas.py (新增)
**功能**: 生成空间关系问答

**问题类型**:
1. 第一人称方向 - 8方向 (12个)
2. 距离比较 - 更远/更近 (10个)
3. 大小比较 - 体积 (10个)

**核心函数**:
- `calculate_object_direction()`: 计算物体相对相机的方向角度
- `get_direction_label()`: 角度转方向标签
- `get_all_valid_directions()`: 获取所有有效方向 (避免歧义)
- `generate_egocentric_question()`: 生成方向问题
- `generate_distance_comparison_question()`: 生成距离比较问题
- `generate_size_comparison_question()`: 生成大小比较问题

**8方向定义**:
```python
DIRECTION_DEFINITIONS = {
    "front": (-30, 30),
    "front-right": (15, 75),
    "right": (45, 135),
    "back-right": (105, 165),
    "back": (135, -135),  # 特殊: 环绕±180
    "back-left": (-165, -105),
    "left": (-135, -45),
    "front-left": (-75, -15),
}
```

**歧义处理**:
- 如果物体在边界 (如30°,同时符合"front"和"front-right")
- 优先选择4方向 (front/right/back/left)
- 干扰选项排除所有有效方向

### 4. CLAUDE.md (更新)
**新增章节**:
- 7. 度量测量问答 (第114-134行)
- 8. 空间关系问答 (第136-158行)
- 更新帧metadata说明 (第188-195行)
- 更新输出路径 (第219-220行)

### 5. README_NEW_TASKS.md (新增)
**内容**:
- 详细使用指南
- 参数说明
- 问题示例
- 配置选项
- 完整使用流程
- 故障排查

### 6. IMPLEMENTATION_SUMMARY.md (本文件)

## 技术实现细节

### 距离计算
```python
def euclidean_distance_3d(pos1, pos2):
    dx = pos1["x"] - pos2["x"]
    dy = pos1["y"] - pos2["y"]
    dz = pos1["z"] - pos2["z"]
    return math.sqrt(dx*dx + dy*dy + dz*dz)
```

### 方向计算
```python
def calculate_object_direction(camera_pos, camera_yaw, object_pos):
    # 1. 计算相机到物体的向量 (XZ平面)
    dx = object_pos["x"] - camera_pos["x"]
    dz = object_pos["z"] - camera_pos["z"]

    # 2. 计算绝对角度
    obj_angle = math.degrees(math.atan2(dx, dz))

    # 3. 减去相机yaw得到相对角度
    camera_forward_angle = -camera_yaw
    relative_angle = normalize_angle(obj_angle - camera_forward_angle)

    return relative_angle  # [-180, 180]
```

### 体积计算
```python
def calculate_object_volume(bbox):
    size = bbox.get("size", {})
    width = abs(size.get("x", 0))
    height = abs(size.get("y", 0))
    depth = abs(size.get("z", 0))
    return width * height * depth
```

## 数据流

```
executable_plan.py
    ↓ 执行任务,录制帧
outputs/video/<episode>/agent_1/
    ├── img_00000.png
    ├── img_00000.json  ← 包含详细物体信息和相机参数
    ├── img_00001.png
    ├── img_00001.json
    └── ...
    ↓
generate_metric_qas.py         generate_spatial_qas.py
    ↓                               ↓
outputs/metric/                outputs/spatial/
  └── metric_questions.json      └── spatial_questions.json
```

## 输出示例

### 度量测量问题
```json
{
  "question": "Calculate the distance from the camera to the Apple.",
  "options": ["145.23 cm", "210.45 cm", "98.76 cm", "182.34 cm"],
  "answer": "145.23 cm",
  "metadata": {
    "type": "camera_to_object_distance",
    "object": "Apple",
    "distance_value": 145.23,
    "unit": "cm",
    "frame_index": 42
  }
}
```

### 空间关系问题
```json
{
  "question": "From the camera's perspective, in which direction is the Apple?",
  "options": ["front-right", "back", "left", "front-left"],
  "answer": "front-right",
  "metadata": {
    "type": "egocentric_direction",
    "object": "Apple",
    "direction": "front-right",
    "angle": 45.23,
    "all_valid_directions": ["front-right", "right"],
    "frame_index": 42
  }
}
```

## 测试验证

已通过单元测试验证以下功能:
- ✓ 3D欧氏距离计算
- ✓ 单位转换 (m → cm)
- ✓ Bounding box尺寸计算
- ✓ 角度归一化
- ✓ 方向计算 (8个方向场景)
- ✓ 旋转相机的方向计算
- ✓ 体积计算
- ✓ 干扰项相对误差 > 20%
- ✓ 边界情况处理

## 配置灵活性

### 单位统一修改
所有距离/尺寸单位集中在脚本顶部:
```python
# generate_metric_qas.py
DISTANCE_UNIT = "cm"  # 修改此处即可统一单位
SIZE_UNIT = "cm"
DECIMAL_PLACES = 2
```

### 方向定义修改
```python
# generate_spatial_qas.py
DIRECTION_DEFINITIONS = {
    "front": (-30, 30),  # 可调整角度范围
    # ...
}
```

### 干扰项误差调整
```python
# generate_metric_qas.py 第239行
error_factor = random.uniform(0.25, 0.8)  # 可调整误差范围
```

## 使用限制

1. **需要Linux环境** - AI2-THOR依赖
2. **需要重新生成数据** - 旧episode数据缺少必要字段
3. **物体可见性要求** - 物体必须在相机视野内
4. **最少物体数要求**:
   - 物体间距离/大小比较: 至少2个可见物体
   - 其他问题: 至少1个可见物体

## 后续扩展建议

### 可能的新问题类型
1. **相对方向** - "从A的视角看,B在哪个方向？"
2. **连续值范围** - 不用多选,直接回答数值
3. **多物体场景** - "场景中最大的3个物体是什么？"
4. **高度比较** - "哪个物体更高？"
5. **面积/周长** - 2D投影尺寸

### 代码优化
1. 缓存帧数据避免重复读取
2. 并行处理多个episode
3. 增加进度条显示
4. 支持批量配置文件

## 兼容性

### 向后兼容
- `executable_plan.py` 修改不影响现有QA生成器
- 旧的 `itemlist` 字段保留
- 新字段为可选,不存在时优雅降级

### 向前兼容
- 单位配置集中,易于修改
- 问题模板独立,易于扩展
- 方向定义可配置

## 文件清单

### 修改的文件
- ✓ `executable_plan.py` (2处修改)
- ✓ `CLAUDE.md` (4处更新)

### 新增的文件
- ✓ `generate_metric_qas.py` (530行)
- ✓ `generate_spatial_qas.py` (515行)
- ✓ `README_NEW_TASKS.md` (详细文档)
- ✓ `IMPLEMENTATION_SUMMARY.md` (本文件)

### 总代码量
- 新增Python代码: ~1045行
- 新增文档: ~600行
- 总计: ~1645行

## 状态
✅ **已完成并通过测试**

可在Linux环境直接使用,Windows环境下代码正常但无法运行 (AI2-THOR限制)。
