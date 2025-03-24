#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Type

from dfpipe.core.base import DataLoader, DataProcessor, DataWriter

logger = logging.getLogger("DataPipeline.Registry")


class ComponentRegistry:
    """组件注册表，用于管理所有注册的加载器、处理器和输出器"""

    _loaders: Dict[str, Type[DataLoader]] = {}
    _processors: Dict[str, Type[DataProcessor]] = {}
    _writers: Dict[str, Type[DataWriter]] = {}

    @classmethod
    def register_loader(cls, loader_class: Type[DataLoader]) -> Type[DataLoader]:
        """
        注册数据加载器类

        Args:
            loader_class: 数据加载器类

        Returns:
            注册的数据加载器类
        """
        cls._loaders[loader_class.__name__] = loader_class
        logger.debug(f"注册加载器: {loader_class.__name__}")
        return loader_class

    @classmethod
    def register_processor(
        cls, processor_class: Type[DataProcessor]
    ) -> Type[DataProcessor]:
        """
        注册数据处理器类

        Args:
            processor_class: 数据处理器类

        Returns:
            注册的数据处理器类
        """
        cls._processors[processor_class.__name__] = processor_class
        logger.debug(f"注册处理器: {processor_class.__name__}")
        return processor_class

    @classmethod
    def register_writer(cls, writer_class: Type[DataWriter]) -> Type[DataWriter]:
        """
        注册数据输出器类

        Args:
            writer_class: 数据输出器类

        Returns:
            注册的数据输出器类
        """
        cls._writers[writer_class.__name__] = writer_class
        logger.debug(f"注册输出器: {writer_class.__name__}")
        return writer_class

    @classmethod
    def get_loader(cls, name: str, **kwargs) -> DataLoader:
        """
        获取数据加载器实例

        Args:
            name: 加载器类名
            **kwargs: 传递给加载器构造函数的参数

        Returns:
            数据加载器实例
        """
        if name not in cls._loaders:
            raise ValueError(f"未知的数据加载器: {name}")
        return cls._loaders[name](**kwargs)

    @classmethod
    def get_processor(cls, name: str, **kwargs) -> DataProcessor:
        """
        获取数据处理器实例

        Args:
            name: 处理器类名
            **kwargs: 传递给处理器构造函数的参数

        Returns:
            数据处理器实例
        """
        if name not in cls._processors:
            raise ValueError(f"未知的数据处理器: {name}")
        return cls._processors[name](**kwargs)

    @classmethod
    def get_writer(cls, name: str, **kwargs) -> DataWriter:
        """
        获取数据输出器实例

        Args:
            name: 输出器类名
            **kwargs: 传递给输出器构造函数的参数

        Returns:
            数据输出器实例
        """
        if name not in cls._writers:
            raise ValueError(f"未知的数据输出器: {name}")
        return cls._writers[name](**kwargs)

    @classmethod
    def list_loaders(cls) -> List[str]:
        """
        列出所有已注册的数据加载器

        Returns:
            数据加载器类名列表
        """
        return list(cls._loaders.keys())

    @classmethod
    def list_processors(cls) -> List[str]:
        """
        列出所有已注册的数据处理器

        Returns:
            数据处理器类名列表
        """
        return list(cls._processors.keys())

    @classmethod
    def list_writers(cls) -> List[str]:
        """
        列出所有已注册的数据输出器

        Returns:
            数据输出器类名列表
        """
        return list(cls._writers.keys())

    @classmethod
    def auto_discover(cls):
        """自动发现并注册所有组件"""
        logger.info("开始自动发现组件...")

        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 发现加载器
        cls._discover_components(
            os.path.join(package_dir, "loaders"), DataLoader, cls.register_loader
        )

        # 发现处理器
        cls._discover_components(
            os.path.join(package_dir, "processors"),
            DataProcessor,
            cls.register_processor,
        )

        # 发现输出器
        cls._discover_components(
            os.path.join(package_dir, "writers"), DataWriter, cls.register_writer
        )

        logger.info(
            f"自动发现完成, 发现 {len(cls._loaders)} 个加载器, {len(cls._processors)} 个处理器, {len(cls._writers)} 个输出器"
        )

    @classmethod
    def _discover_components(cls, directory: str, base_class: Type, register_func):
        """
        发现指定目录下的组件

        Args:
            directory: 组件所在目录
            base_class: 组件基类
            register_func: 注册函数
        """
        if not os.path.exists(directory):
            logger.warning(f"目录不存在: {directory}")
            return

        # 获取目录下所有Python文件
        module_files = [
            f for f in os.listdir(directory) if f.endswith(".py") and f != "__init__.py"
        ]

        for file in module_files:
            module_name = file[:-3]  # 去掉.py后缀
            full_module_name = f"dfpipe.{os.path.basename(directory)}.{module_name}"

            try:
                # 导入模块
                module = importlib.import_module(full_module_name)

                # 查找所有继承自base_class的类
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, base_class)
                        and obj != base_class
                    ):
                        register_func(obj)
                        logger.debug(f"从 {full_module_name} 发现并注册 {name}")

            except Exception as e:
                logger.error(f"导入模块 {full_module_name} 失败: {str(e)}")
