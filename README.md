# DFPipe

[![English](https://img.shields.io/badge/English-007ACC?style=flat-square&logo=python&logoColor=white)](README_EN.md) [![中文](https://img.shields.io/badge/中文-4A4A4A?style=flat-square&logo=python&logoColor=white)](README.md)

一个灵活、可扩展的DataFrame处理管道工具，支持多种数据源、处理算法和输出格式。

## 特点

- **模块化设计**: 提供数据加载器、处理器和输出器三种基础组件
- **灵活配置**: 支持通过JSON配置文件或代码API构建处理流程
- **易于扩展**: 简单的组件注册机制，方便添加自定义组件
- **丰富日志**: 详细的处理日志，便于调试和监控
- **广泛兼容**: 支持 Python 3.6 到 3.12 的所有版本

## 系统要求

- Python >= 3.6
- pandas >= 1.3.0
- numpy >= 1.20.0

## 测试覆盖率

DFPipe 维持高水平的测试覆盖率，确保库的可靠性和稳定性：

- **全面的测试套件**：涵盖所有核心组件的单元测试和端到端工作流的集成测试
- **基于模拟的测试**：使用模拟对象（mock）高效测试文件系统和外部依赖
- **CI/CD集成**：在多个Python版本上自动化测试，确保跨版本兼容性

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=dfpipe

# 生成HTML覆盖率报告
pytest --cov=dfpipe --cov-report=html
```

当前核心组件的测试覆盖率超过99%，确保在各种场景下都能可靠运行。

## 安装

```bash
# 从 PyPI 安装
pip install dfpipe

# 从源码安装
git clone https://github.com/Ciciy-l/dfpipe.git
cd dfpipe
pip install -e .

# 安装开发版本（包含测试工具）
pip install -e ".[dev]"
```

## 快速开始

### 通过命令行使用

最简单的使用方式是通过命令行直接运行：

```bash
python -m dfpipe --input-dir data --output-dir output
```

这将使用默认的CSV加载器和输出器处理数据。

### 使用配置文件

创建配置文件来定义完整的数据处理流程：

```bash
python -m dfpipe --config dfpipe/examples/simple.json
```

### 配置文件示例

```json
{
    "name": "简单数据处理管道",
    "loader": {
        "name": "CSVLoader",
        "params": {
            "input_dir": "data",
            "file_pattern": "*.csv"
        }
    },
    "processors": [
        {
            "name": "FilterProcessor",
            "params": {
                "column": "age",
                "condition": 18
            }
        }
    ],
    "writer": {
        "name": "CSVWriter",
        "params": {
            "output_dir": "output",
            "filename": "processed_data.csv"
        }
    }
}
```

## 通过代码使用

可以在Python代码中使用API构建和执行管道：

```python
from dfpipe import Pipeline, ComponentRegistry, setup_logging

# 设置日志
logger = setup_logging()

# 初始化组件注册表
ComponentRegistry.auto_discover()

# 创建管道
pipeline = Pipeline(name="MyPipeline")

# 设置加载器
loader = ComponentRegistry.get_loader("CSVLoader", input_dir="data")
pipeline.set_loader(loader)

# 添加处理器
filter_processor = ComponentRegistry.get_processor("FilterProcessor", column="age", condition=18)
pipeline.add_processor(filter_processor)

# 设置输出器
writer = ComponentRegistry.get_writer("CSVWriter", output_dir="output")
pipeline.set_writer(writer)

# 运行管道
pipeline.run()
```

## 组件介绍

### 数据加载器

数据加载器负责从各种数据源加载数据，默认提供`CSVLoader`。

#### 内置加载器

- **CSVLoader**: 从CSV文件加载数据
  - `input_dir`: 输入目录
  - `file_pattern`: 文件匹配模式
  - `encoding`: 文件编码

### 数据处理器

数据处理器负责对数据进行处理和转换。

#### 内置处理器

- **FilterProcessor**: 根据条件过滤数据
  - `column`: 列名
  - `condition`: 过滤条件

- **TransformProcessor**: 对列应用转换函数
  - `column`: 要转换的列
  - `transform_func`: 转换函数
  - `target_column`: 结果存储列

- **ColumnProcessor**: 列操作（添加、删除、重命名）
  - `operation`: 操作类型（'add', 'drop', 'rename'）
  - 特定操作的参数

- **FieldsOrganizer**: 组织和规范字段
  - `target_columns`: 目标字段列表，结果将只包含这些字段并按此顺序排列
  - `default_values`: 字段默认值字典，键为字段名，值为默认值。如果不提供，默认使用空字符串

### 数据输出器

数据输出器负责将处理后的数据保存到各种目标位置。

#### 内置输出器

- **CSVWriter**: 将数据保存为CSV文件
  - `output_dir`: 输出目录
  - `filename`: 文件名
  - `encoding`: 文件编码

## 自定义组件

### 创建自定义加载器

```python
from dfpipe import DataLoader, ComponentRegistry

@ComponentRegistry.register_loader
class MyCustomLoader(DataLoader):
    def __init__(self, param1, param2=None, **kwargs):
        super().__init__(
            name="MyCustomLoader",
            description="我的自定义加载器"
        )
        self.param1 = param1
        self.param2 = param2

    def load(self) -> pd.DataFrame:
        # 实现加载逻辑
        # ...
        return data_frame
```

### 创建自定义处理器

```python
from dfpipe import DataProcessor, ComponentRegistry

@ComponentRegistry.register_processor
class MyCustomProcessor(DataProcessor):
    def __init__(self, param1, param2=None, **kwargs):
        super().__init__(
            name="MyCustomProcessor",
            description="我的自定义处理器"
        )
        self.param1 = param1
        self.param2 = param2

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # 实现处理逻辑
        # ...
        return processed_data
```

### 创建自定义输出器

```python
from dfpipe import DataWriter, ComponentRegistry

@ComponentRegistry.register_writer
class MyCustomWriter(DataWriter):
    def __init__(self, param1, param2=None, **kwargs):
        super().__init__(
            name="MyCustomWriter",
            description="我的自定义输出器"
        )
        self.param1 = param1
        self.param2 = param2

    def write(self, data: pd.DataFrame) -> None:
        # 实现输出逻辑
        # ...
```

## 许可证

MIT

## 开发者指南

### 代码风格规范

本项目使用以下工具维护代码风格一致性：

- **Black**: 自动代码格式化工具
- **isort**: 导入语句排序工具

#### 本地配置

1. 安装开发依赖：

```bash
pip install black isort pre-commit
```

2. 设置pre-commit钩子：

```bash
pre-commit install
```

这将在每次提交前自动运行代码格式化检查。

#### 手动格式化

```bash
# 格式化所有Python文件
black .

# 排序所有Python文件中的导入
isort .
```

#### 配置文件

代码风格配置在`pyproject.toml`文件中定义。

### 持续集成

我们使用GitHub Actions进行持续集成，确保所有提交都通过测试和代码风格检查。workflow配置见`.github/workflows/python-test.yml`。
