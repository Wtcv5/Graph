# Tunnel Geology Modeler — 用户使用文档

隧道超前地质预报 3D 建模工具。将地震波速 (Vp/Vs) 扫描线数据转化为可计算的地质模型，支持参数化隧道构建与围岩耦合分析。

## 目录

1. [环境准备](#1-环境准备)
2. [数据预处理](#2-数据预处理)
3. [GUI 操作指南](#3-gui-操作指南)
4. [Python API 编程接口](#4-python-api-编程接口)
5. [隧道建模工作流](#5-隧道建模工作流)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 系统要求

- Windows 10/11, macOS 12+, 或 Linux
- Python 3.10+
- 8GB+ RAM（处理 16M 点数据推荐 16GB）

### 安装

```powershell
# 克隆仓库
git clone https://github.com/Wtcv5/Graph.git
cd Graph

# 创建虚拟环境
python -m venv .venv

# 安装依赖
.\.venv\Scripts\python -m pip install -r requirements.txt

# 安装项目包（可选，提供 CLI 入口）
.\.venv\Scripts\python -m pip install -e .
```

### 验证安装

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
# 应输出: Ran XX tests ... OK
```

---

## 2. 数据预处理

### 输入格式

两个 CSV 文件，无表头，4 列：

| 列 | 含义 | 示例 |
|----|------|------|
| X | 纵向坐标 (m) | -12359.640 |
| Y | 横向坐标 (m) | 50.000 |
| Z | 高程 (m) | 2975.818 |
| Velocity | 波速 (m/s) | 5970.088 |

### 方式 A：命令行

```powershell
# 使用默认配置
.\.venv\Scripts\python scripts/preprocess.py --force

# 指定配置文件
.\.venv\Scripts\python scripts/preprocess.py --config configs/datasets/my_dataset.json --force

# 覆盖输出目录
.\.venv\Scripts\python scripts/preprocess.py --output-dir outputs/my_project --force
```

### 方式 B：配置文件

创建 `configs/datasets/my_dataset.json`：

```json
{
  "name": "my_dataset",
  "description": "My tunnel Vp/Vs data",
  "data_dir": "path/to/data",
  "vp_file": "Vp.csv",
  "vs_file": "Vs.csv",
  "output_dir": "outputs/my_dataset",
  "grid_step_m": 0.5,
  "chunk_size": 500000,
  "fill_kernel_size": 3,
  "fill_max_iter": 5,
  "title": "My Geological Model",
  "source": "My Project",
  "coordinate_system": "Local engineering coordinates (meters)",
  "density_formula": "Gardner: rho = 0.31 * Vp^0.25 (g/cm3)"
}
```

### 输出

```
outputs/<dataset_name>/
├── geological_model.nc      # NetCDF 规则 3D 网格 (~550 MB)
└── preprocessing_info.json  # 处理元数据
```

---

## 3. GUI 操作指南

### 启动

```powershell
# 方式 1：直接启动
.\.venv\Scripts\python scripts/run_gui.py

# 方式 2：启动时加载数据
.\.venv\Scripts\python scripts/run_gui.py outputs/right_tunnel_exit_vpvs/geological_model.nc

# 方式 3：CLI 入口（需 pip install -e .）
.\.venv\Scripts\tunnel-velocity-model gui
```

### 界面布局

```
┌──────────────┬─────────────────────────┬────────────────┐
│ 📁 Data      │                        │ 📊 Properties  │
│ (数据面板)    │     3D 视图             │ (属性面板)      │
│              │                        │                │
├──────────────│                        ├────────────────┤
│ 🚇 Tunnel    │                        │ ✂️ Slices      │
│ (隧道面板)    │                        │ (切片+等值面)   │
│              │                        │                │
│              ├────────────────────────┤                │
│              │  📋 Log (日志面板)       │                │
└──────────────┴─────────────────────────┴────────────────┘
```

### 各面板功能

#### 📁 Data — 数据面板

- **Open NetCDF** — 加载地质模型文件
- **Fields** — 点击切换显示的属性场（Vp, Vs, Density 等）
- **Classification** — 点击切换到分类场（BQ, Poisson_Ratio 等）

#### ✂️ Slices — 切片面板

**正交切片**:
- 拖动 X / Y / Z 滑块在三个方向上切剖面
- 实时更新，展示该切面的属性分布

**等值面** (Isosurface):
- 输入阈值（如 Vp=5800 m/s）
- 点击 **Build** 生成 3D 等值面（半透明金色）
- 点击 **Clear** 移除等值面
- ⚠️ 大模型计算约需 3-10 秒

#### 🚇 Tunnel — 隧道面板

用于构建参数化隧道并与地质模型耦合。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| X start | 隧道起点 X 坐标 | 模型前边界 + 50m |
| Y start | 隧道起点 Y 坐标 | 0 |
| Z start | 隧道起点高程 | 模型中间 |
| Length | 隧道长度 (m) | 180 |
| Gradient | 坡度 (%) | 0.5 |
| Shape | 断面形状 | horseshoe |
| Width | 开挖宽度 (m) | 12 |
| Height | 开挖高度 (m) | 10 |
| Shotcrete | 喷射混凝土厚度 (m) | 0.15 |
| Lining | 二次衬砌厚度 (m) | 0.35 |

操作流程：
1. 调整参数 → 点击 **🔨 Build Tunnel**
2. 隧道 3D 网格出现在视图中
3. 点击 **🔗 Couple with Geology** 计算围岩属性
4. 结果面板显示分段围岩分级

#### 📊 Properties — 属性面板

- **Field Statistics** — 当前属性场的 min/max/mean/std
- **Probe Point** — 在 3D 视图中点击任意位置，显示该点的全部属性值
- **Rock Mass Class** — 显示拾取点的 BQ 分级（I~V）

### 鼠标与快捷键

| 操作 | 效果 |
|------|------|
| 左键拖拽 | 旋转视角 |
| 右键拖拽 / 滚轮 | 缩放 |
| 中键拖拽 | 平移 |
| 左键点击模型 | 拾取点属性 |
| `R` | 重置视角 |
| `T` | 俯视图 |
| `F` | 正视图 |
| `S` | 侧视图 |

### 导出

- **File → Export VTK** — 导出为 VTK 格式（可在 ParaView 中打开）
- **File → Export XYZ** — 导出为 CSV 点云（10% 采样）

---

## 4. Python API 编程接口

### 加载模型

```python
import sys
sys.path.insert(0, 'src')
from tunnel_geology_model import load_from_netcdf

model = load_from_netcdf('outputs/right_tunnel_exit_vpvs/geological_model.nc')

# 基本信息
print(model.summary())
print(f"Shape: {model.shape}")      # (201, 406, 211)
print(f"Vp range: {model.stats('Vp')}")
```

### 岩体分类

```python
from tunnel_geology_model import classify_bq, classify_rmr

# BQ 国标分级
bq_index, bq_class = classify_bq(model['Vp'])

# 添加到模型
model.classification['BQ'] = bq_index
model.classification['BQ_Class'] = bq_class.astype('float32')
```

### 剖面提取

```python
from tunnel_geology_model import (
    extract_xz_section,     # 横截面（垂直隧道）
    extract_yz_section,     # 纵截面（沿隧道）
    extract_xy_section,     # 水平截面
    extract_arbitrary_section,  # 任意方向
)

# 在 Y=0 处切横截面
xz = extract_xz_section(model, 0.0, field_names=['Vp', 'Vs'])
vp_cross = xz['Vp']  # shape: (406, 211)
```

### 等值面

```python
from tunnel_geology_model import marching_cubes_isosurface

verts, faces, normals, values = marching_cubes_isosurface(
    model['Vp'],
    iso_value=5800,
    spatial_step=model.grid_step,
    origin=(model.x_range[0], model.y_range[0], model.z_range[0]),
)
# verts: (N, 3) 顶点坐标
# faces: (M, 3) 三角形索引
```

### 导出

```python
from tunnel_geology_model import (
    to_vtk_rectilinear_grid,  # VTK RectilinearGrid (.vtr)
    to_vtk_structured_grid,   # VTK StructuredGrid (.vts)
    to_numpy_dict,            # NumPy .npz 压缩包
    to_xyz_point_cloud,       # CSV 点云
)

to_vtk_rectilinear_grid(model, 'output.vtr')
to_xyz_point_cloud(model, 'points.csv', sample=0.1)  # 10% 采样
```

### 统计分析

```python
from tunnel_geology_model import (
    field_histogram,
    depth_profile,
    anomaly_detection,
    correlation_map,
)

# Vp 随深度变化
z, mean, std = depth_profile(model, 'Vp')

# Vp/Vs 相关性
r = correlation_map(model, 'Vp', 'Vs')

# 异常检测（>2σ）
anomalies = anomaly_detection(model, 'Vp', n_sigma=2.0)
```

### 隧道建模

```python
from tunnel_geology_model import (
    ParametricTunnel,
    TunnelAlignment,
    TunnelSection,
    SectionShape,
)

# 1. 定义中心线
alignment = TunnelAlignment.from_endpoints(
    x_start=-12300, y_start=0, z_start=3000,
    x_end=-12170, y_end=0, z_end=3020,
    gradient_pct=1.0,
)

# 2. 定义断面
section = TunnelSection(
    shape=SectionShape.HORSESHOE,
    width=12, height=10,
    shotcrete_thickness=0.15,
    lining_thickness=0.35,
)

# 3. 构建隧道
tunnel = ParametricTunnel(alignment=alignment, section=section, name='Main_Tunnel')
tunnel.build_mesh()
print(tunnel.summary())
```

### 隧道-地质耦合

```python
from tunnel_geology_model import TunnelGeologyCoupling

coupling = TunnelGeologyCoupling(tunnel, model)

# 隧道表面地质属性
props = coupling.compute_vertex_properties()
print(coupling.vertex_statistics())

# 沿隧道中心线剖面
profile = coupling.compute_centerline_profile()

# 分段围岩分类（每20m一段）
segments = coupling.classify_chainage_segments(segment_length=20)
for seg in segments:
    print(f"[{seg['chainage_start']:.0f}-{seg['chainage_end']:.0f}]m "
          f"BQ={seg.get('mean_BQ', 0):.0f} "
          f"({seg.get('BQ_class', '?')}级)")

# 提取隧道周边区域
zone = coupling.extract_excavation_zone(margin=3.0)
```

---

## 5. 隧道建模工作流

### 完整流程

```
原始 CSV → 预处理管线 → NetCDF 网格 → 加载 GUI/Python
                                         │
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                      3D 可视化      隧道建模        数据分析
                      (切片/等值面)   (参数化+耦合)    (统计/导出)
```

### 典型工作流示例

1. **数据导入**：
   ```powershell
   .venv\Scripts\python scripts/preprocess.py --force
   ```

2. **可视化探索**：
   启动 GUI，拖动切片浏览速度场变化，构建等值面识别高速/低速异常

3. **隧道设计**：
   在 Tunnel 面板设置隧道参数 → Build → Couple

4. **围岩评价**：
   查看分段 BQ 分级，识别 IV 级软弱段

5. **导出报告**：
   File → Export VTK → 在 ParaView 中做精细化后处理

---

## 6. 常见问题

### Q: GUI 启动后窗口无响应？
A: 等待 5-10 秒加载数据。如果持续无响应，可能是 VTK/OpenGL 驱动问题，尝试更新显卡驱动。

### Q: 等值面 Build 很慢？
A: 17M 单元格的 marching cubes 计算量大（3-10 秒），属正常现象。可以先 cut 子区域再操作：
```python
subset = model.subset_x(-12300, -12200)
```

### Q: 如何增加新的数据源？
A: 创建新的 JSON 配置文件，放在 `configs/datasets/` 目录下，指定新的 CSV 路径即可。

### Q: 输出文件很大？
A: NetCDF 使用 zlib 压缩。如需更小体积，可以用 `to_numpy_dict()` 导出 `.npz` 格式。

### Q: 如何在 ParaView 中查看？
A: File → Export VTK → 选择 `.vtr` 格式 → 在 ParaView 中 File → Open。

### Q: 中文路径报错？
A: 使用 `engine='h5netcdf'` 读写 NetCDF 文件（已默认配置）。VTK 导出和导入也应使用英文路径。
