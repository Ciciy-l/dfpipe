#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class DataLoader(ABC):
    """
    数据加载器基类

    所有数据加载器必须继承此类并实现load方法
    """

    def __init__(self, name: str, description: str, **kwargs):
        """
        初始化数据加载器

        Args:
            name: 加载器名称
            description: 加载器描述
            **kwargs: 其他参数
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"DataPipeline.Loader.{name}")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        从数据源加载数据

        Returns:
            加载的数据框
        """
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class DataProcessor(ABC):
    """
    数据处理器基类

    所有数据处理算法必须继承此类并实现process方法
    """

    def __init__(self, name: str, description: str, **kwargs):
        """
        初始化数据处理器

        Args:
            name: 处理器名称
            description: 处理器描述
            **kwargs: 其他参数
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"DataPipeline.Processor.{name}")

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理数据

        Args:
            data: 输入数据框

        Returns:
            处理后的数据框
        """
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class DataWriter(ABC):
    """
    数据输出器基类

    所有数据输出器必须继承此类并实现write方法
    """

    def __init__(self, name: str, description: str, **kwargs):
        """
        初始化数据输出器

        Args:
            name: 输出器名称
            description: 输出器描述
            **kwargs: 其他参数
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"DataPipeline.Writer.{name}")

    @abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """
        将数据输出到目标位置

        Args:
            data: 要输出的数据框
        """
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
