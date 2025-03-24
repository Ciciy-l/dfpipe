#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据流水线处理工具包

这个包提供了灵活的数据加载、处理和输出功能，支持通过配置文件或代码构建数据处理流水线。
"""

__version__ = "0.0.3"
__author__ = "Leyia"

# 导出公共接口
from dfpipe.core.base import DataLoader, DataProcessor, DataWriter
from dfpipe.core.pipeline import Pipeline
from dfpipe.core.registry import ComponentRegistry

# 导入内置组件，确保它们被注册
from dfpipe.loaders.csv_loader import CSVLoader
from dfpipe.processors.base_processor import (
    ColumnProcessor,
    FieldsOrganizer,
    FilterProcessor,
    TransformProcessor,
)
from dfpipe.utils.logging import setup_logging
from dfpipe.writers.csv_writer import CSVWriter

# 为常用类提供简短别名
Loader = DataLoader
Processor = DataProcessor
Writer = DataWriter
Registry = ComponentRegistry

# 自动执行组件发现
ComponentRegistry.auto_discover()
