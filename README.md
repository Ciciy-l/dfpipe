# DFPipe

一个灵活、可扩展的DataFrame处理管道工具，支持多种数据源、处理算法和输出格式。

## 特点

- **模块化设计**: 提供数据加载器、处理器和输出器三种基础组件
- **灵活配置**: 支持通过JSON配置文件或代码API构建处理流程
- **易于扩展**: 简单的组件注册机制，方便添加自定义组件
- **丰富日志**: 详细的处理日志，便于调试和监控

## 安装

```bash
# 从 PyPI 安装
pip install dfpipe

# 从源码安装
git clone https://github.com/Ciciy-l/dfpipe.git
cd dfpipe
pip install -e .
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