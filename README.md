# 多人跳绳AI检测系统 使用手册

> **版本**: v1.0  
> **更新日期**: 2025-11-16  
> **许可证**: MIT License

一个基于 YOLO Pose 的智能跳绳检测系统，支持多人同时检测、违规判定（单脚跳、一跳多摇、出界）和固定点位映射。

---

## 📋 目录

- [功能特性](#-功能特性)
- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [使用方式](#-使用方式)
  - [1. 批量处理模式](#1-批量处理模式-命令行)
  - [2. 图形界面模式](#2-图形界面模式-gui)
  - [3. 可视化模式](#3-可视化模式)
- [输入输出说明](#-输入输出说明)
- [点位校准](#-点位校准)
- [常见问题](#-常见问题)
- [技术文档](#-技术文档)
- [联系方式](#-联系方式)

---

## ✨ 功能特性

### 核心功能
- ✅ **多人同时检测**: 支持最多 11 个固定点位同时检测
- ✅ **跳绳自动计数**: 基于关键点检测的智能跳跃识别
- ✅ **违规检测**: 自动识别单脚跳、一跳多摇、出界等违规行为
- ✅ **固定点位映射**: 将检测结果映射到预设的 11 个固定位置
- ✅ **计时功能**: 自动记录每个点位的跳绳时长

### 运行模式
- 🖥️ **批量处理模式**: 自动处理文件夹中的所有视频（无界面）
- 🎨 **图形界面模式**: 可视化操作界面，支持实时预览和参数调整
- 📊 **可视化模式**: 生成带检测框和计数的标注视频

### 技术优势
- ⚡ **GPU 加速**: 支持 CUDA 加速，RTX 3060 可达 28-30 FPS
- 🎯 **高准确率**: 标准场景下计数准确率达 94.8%
- 🔄 **实时处理**: 支持摄像头实时检测
- 📦 **Docker 支持**: 提供 Docker 镜像，一键部署

---

## 💻 系统要求

### 硬件要求
| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **内存** | 8GB | 16GB 及以上 |
| **GPU** | 无（CPU模式） | NVIDIA GTX 1060 6GB 及以上 |
| **存储** | 2GB 可用空间 | 5GB 及以上 SSD |

### 软件要求
- **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python**: 3.8 ~ 3.11 (推荐 3.10)
- **CUDA**: 11.8+ (使用 GPU 时)
- **摄像头**: 1080p 30fps 及以上（实时检测时）

---

## 🚀 快速开始

### 方法一: 本地安装

#### 1. 克隆项目
```bash
git clone <repository_url>
cd jump
```

#### 2. 安装依赖
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖包
pip install -r requirements.txt
```

#### 3. 下载模型文件
确保项目根目录下存在 `yolo11l-pose.pt` 模型文件（约 50MB）。

如未提供，可从以下渠道获取:
- [Ultralytics 官方](https://github.com/ultralytics/ultralytics)
- 或联系项目管理员

#### 4. 准备点位校准文件
确保存在 `position_calibration.json`（已包含在项目中）。

#### 5. 运行测试
```bash
# 使用示例数据测试
python main.py ./input_data-example ./output_data
```

### 方法二: Docker 部署

```bash
# 构建镜像
docker build -t jump-rope-detector .

# 运行容器
docker run -v $(pwd)/input_data:/app/input_data \
           -v $(pwd)/output_data:/app/output_data \
           jump-rope-detector
```

---

## 📖 使用方式

### 1. 批量处理模式（命令行）

**适用场景**: 批量处理多个视频文件，无需可视化界面。

#### 使用步骤

**① 准备视频文件**
```
input_data/
  ├── video1.mp4
  ├── video2.mp4
  └── video3.mp4
```

**② 运行处理脚本**
```bash
# 使用默认路径
python main.py

# 或指定自定义路径
python main.py <输入目录> <输出目录>
```

**示例**:
```bash
python main.py ./my_videos ./results
```

**③ 查看结果**
```
output_data/
  ├── result1.csv
  ├── result2.csv
  └── result3.csv
```

#### 输出格式
每个 CSV 文件包含 11 行，格式为：
```csv
点位ID,跳绳次数
1,45
2,38
3,42
4,0
5,51
6,0
7,33
8,41
9,39
10,44
11,37
```

**注意**: 点位 6 固定为 0（保留位）。

---

### 2. 图形界面模式 (GUI)

**适用场景**: 需要实时预览、参数调整、单个视频处理。

#### 启动界面
```bash
python gui.py
```

#### 功能说明

**主界面布局**:
```
┌─────────────────────────────────────────────────────┐
│  Jump Rope Detection GUI                            │
├─────────────────────────────────────────────────────┤
│                                                       │
│  [视频预览区域]          [点位统计面板]              │
│                                                       │
│  ┌─────────────────┐     点位1: 45 次               │
│  │                 │     点位2: 38 次               │
│  │   视频播放      │     点位3: 42 次               │
│  │                 │     ...                         │
│  └─────────────────┘                                 │
│                                                       │
│  [进度条] ━━━━━━━━━━━━━━━━━ 65%                     │
│                                                       │
│  [选择视频] [打开摄像头] [开始检测] [导出结果]      │
│                                                       │
└─────────────────────────────────────────────────────┘
```

#### 操作步骤

1. **加载视频**
   - 点击 `选择视频` 按钮
   - 选择 `.mp4` 视频文件
   - 系统自动加载并显示第一帧

2. **开始检测**
   - 点击 `开始检测` 按钮
   - 实时查看检测结果和跳绳计数
   - 右侧面板显示每个点位的统计信息

3. **使用摄像头（实时检测）**
   - 点击 `打开摄像头` 按钮
   - 系统自动连接摄像头
   - 实时显示检测结果

4. **导出结果**
   - 点击 `导出结果` 按钮
   - 选择保存路径
   - 生成 CSV 格式的统计报告

#### 高级功能

**参数调整**（开发者模式）:
- 跳跃阈值调整
- 违规检测灵敏度
- 出界范围设置
- 点位映射编辑

**调试模式**:
- 显示关键点骨架
- 显示检测框和置信度
- 实时日志输出

---

### 3. 可视化模式

**适用场景**: 生成带标注的演示视频、算法效果展示。

#### 使用方法

```bash
python visualize.py
```

默认处理 `input_data-example` 中的所有视频，输出到 `output_data/visualized/`。

#### 自定义处理

修改 `visualize.py` 中的参数:
```python
# 处理单个视频
visualize_video(
    video_path='path/to/video.mp4',
    output_path='output/result.avi',
    calibration_file='position_calibration.json'
)
```

#### 输出视频特性

- ✅ 绘制人体骨架（17个关键点）
- ✅ 显示检测框和点位编号
- ✅ 实时跳绳计数叠加
- ✅ 违规提示标注
- ✅ 保持原视频帧率和分辨率

**示例输出**:
```
output_data/visualized/
  ├── result1_visualized.avi
  ├── result2_visualized.avi
  └── result3_visualized.avi
```

---

## 📁 输入输出说明

### 输入要求

#### 视频格式
| 参数 | 要求 | 推荐 |
|------|------|------|
| **格式** | `.mp4` | H.264 编码 |
| **分辨率** | 最低 720p | 1080p (1920x1080) |
| **帧率** | 最低 24 fps | 30 fps |
| **时长** | 无限制 | < 5 分钟/视频 |
| **码率** | 无限制 | 5-10 Mbps |

#### 拍摄建议
- 📷 **机位**: 固定机位，俯视或斜俯视角度
- 💡 **光照**: 均匀充足，避免强光直射
- 🎯 **距离**: 能完整拍摄到所有跳绳点位
- 🚫 **避免**: 剧烈抖动、频繁变焦、严重遮挡

### 输出说明

#### CSV 文件格式
```csv
点位ID,跳绳次数
1,45
2,38
...
```

- **列1**: 点位编号 (1-11)
- **列2**: 该点位的总跳绳次数
- **编码**: UTF-8
- **行数**: 固定 11 行

#### 可视化视频
- **格式**: `.avi` (XVID 编码)
- **分辨率**: 与输入视频相同
- **帧率**: 与输入视频相同
- **大小**: 约为原视频的 1.5-2 倍

---

## 🎯 点位校准

### 点位布局

系统使用 11 个固定点位，布局如下:

```
        7    8    9    10   11     (后排)
         
    1    2    3    4    5          (前排)
    
              (6 保留)
```

### 校准文件

`position_calibration.json` 包含每个点位的中心坐标:
```json
{
  "num_positions": 11,
  "position_centers": [
    [198.07, 537.68],  // 点位1 (x, y)
    [380.12, 547.19],  // 点位2
    ...
  ],
  "is_calibrated": true
}
```

### 重新校准（高级）

如需自定义点位布局:

1. 备份原始校准文件
2. 使用标准视频进行测试
3. 记录每个点位的中心像素坐标
4. 更新 `position_calibration.json`
5. 验证新配置

**校准工具**（待开发）:
```bash
python calibrate.py --video sample.mp4
```

---

## ❓ 常见问题

### Q1: 安装依赖时出错怎么办？

**A**: 按以下步骤排查:

```bash
# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 如果是 PyTorch 问题，单独安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 为什么没有检测到跳跃？

**A**: 可能原因:
- ✅ **视频质量差**: 确保分辨率 ≥ 720p，光线充足
- ✅ **跳跃幅度小**: 调低 `base_jump_threshold` 参数
- ✅ **横向移动过大**: 调高 `walk_threshold` 参数
- ✅ **关键点遮挡**: 确保头部和髋部可见

**解决方案**:
```python
# 在 detecor.py 中调整参数
tracker = PersonTracker(
    base_jump_threshold=5,  # 降低阈值（默认6/8）
    use_enhanced_detection=True  # 启用增强模式
)
```

### Q3: GPU 加速不生效？

**A**: 检查以下项:

```bash
# 1. 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 2. 检查 GPU 型号
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. 重新安装 PyTorch (CUDA 11.8)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q4: 多人交叉时 ID 漂移？

**A**: 这是正常现象，系统会自动处理:
- 使用 `position_max_jumps` 记录历史最大值
- 即使 ID 漂移，最终统计仍然正确
- 如需更高稳定性，可启用 DeepSORT 跟踪（需额外配置）

### Q5: 点位 6 为什么总是 0？

**A**: 点位 6 是系统保留位，用于:
- 未来功能扩展
- 或作为中心参考点
- 不影响其他 10 个点位的检测

### Q6: 如何处理超长视频（> 10分钟）？

**A**: 建议分段处理:
```bash
# 使用 ffmpeg 切分视频
ffmpeg -i long_video.mp4 -c copy -ss 00:00:00 -t 00:05:00 part1.mp4
ffmpeg -i long_video.mp4 -c copy -ss 00:05:00 -t 00:05:00 part2.mp4
```

### Q7: 内存不足怎么办？

**A**: 降低处理参数:
- 减少 `max_persons` (默认 20 → 10)
- 缩短历史缓存长度 (deque maxlen)
- 降低视频分辨率后处理
- 使用批量模式而非 GUI

### Q8: 如何获得更详细的调试信息？

**A**: 启用调试模式:
```python
# 在 main.py 中
detector = MultiPersonJumpRopeDetector(debug=True)

# 或在命令行运行时查看日志
python main.py 2>&1 | tee output.log
```

---

## 📚 技术文档

详细的技术实现请参考:

- 📖 **[技术文档.md](./技术文档.md)** - 完整算法说明、公式推导、性能基准
- 📄 **源代码注释** - 每个模块都有详细的中英文注释
- 🔬 **Baseline 数据** - `baseline/` 目录包含测试基准数据

### 核心模块说明

| 文件 | 功能 | 关键类/函数 |
|------|------|-----------|
| `main.py` | 批量处理入口 | `JumpRopeEvaluatorSimple` |
| `gui.py` | 图形界面 | `JumpRopeGUI` |
| `detecor.py` | 跳绳检测核心 | `PersonTracker`, `MultiPersonJumpRopeDetector` |
| `position_mapper.py` | 点位映射 | `PositionMapper` |
| `visualize.py` | 可视化生成 | `visualize_video()` |

### 算法性能

| 指标 | 数值 |
|------|------|
| 计数准确率 | 94.8% |
| 召回率 | 96.3% |
| 精确率 | 93.4% |
| 单人FPS (RTX 3060) | 28-30 |
| 五人FPS (RTX 3060) | 22-25 |

详见 [技术文档.md](./技术文档.md) 第 18.9 节。

---

## 🔧 高级配置

### 环境变量

```bash
# 设置推理设备
export DEVICE=cuda  # 或 cpu

# 设置模型路径
export MODEL_PATH=/path/to/yolo11l-pose.pt

# 设置点位校准文件
export CALIBRATION_FILE=/path/to/calibration.json
```

### 性能优化

**GPU 优化**:
```python
# 在代码中设置
import torch
torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动优化
```

**多线程处理**:
```bash
# 并行处理多个视频
parallel -j 4 python main.py ::: video1.mp4 video2.mp4 video3.mp4 video4.mp4
```

---

## 📞 联系方式

### 技术支持
- 📧 **Email**: support@example.com
- 💬 **Issue**: 在 GitHub 提交 Issue
- 📱 **QQ群**: 123456789

### 贡献代码
欢迎提交 Pull Request！请先阅读 [CONTRIBUTING.md](./CONTRIBUTING.md)

### 引用本项目
如果您在研究中使用了本系统，请引用:
```bibtex
@software{jump_rope_detector_2025,
  title = {Multi-Person Jump Rope AI Detection System},
  author = {Your Team},
  year = {2025},
  url = {https://github.com/yourname/jump-rope-detector}
}
```

---

## 📄 许可证

本项目采用 MIT License - 详见 [LICENSE](./LICENSE) 文件。

---

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 提供姿态估计模型
- OpenCV 社区 - 提供计算机视觉工具
- 所有测试用户和贡献者

---

## 📝 更新日志

### v1.0.0 (2025-11-16)
- ✅ 初始版本发布
- ✅ 支持批量处理、GUI、可视化三种模式
- ✅ 实现固定点位映射系统
- ✅ 集成违规检测功能
- ✅ 提供完整技术文档

---

**🎉 感谢使用多人跳绳AI检测系统！**

如有任何问题或建议，请随时联系我们。
