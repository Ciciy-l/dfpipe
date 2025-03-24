#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from dfpipe.core.base import DataLoader, DataProcessor, DataWriter


class Pipeline:
    """
    数据处理管道

    组织和执行数据加载、处理和输出的流程
    """

    def __init__(self, name: str = "DefaultPipeline"):
        """
        初始化数据处理管道

        Args:
            name: 管道名称
        """
        self.name = name
        self.loader = None
        self.processors = []
        self.writer = None
        self.logger = logging.getLogger(f"DataPipeline.Pipeline.{name}")

    def set_loader(self, loader: DataLoader) -> "Pipeline":
        """
        设置数据加载器

        Args:
            loader: 数据加载器实例

        Returns:
            管道实例，支持链式调用
        """
        self.loader = loader
        self.logger.info(f"设置数据加载器: {loader.name}")
        return self

    def add_processor(self, processor: DataProcessor) -> "Pipeline":
        """
        添加数据处理器

        Args:
            processor: 数据处理器实例

        Returns:
            管道实例，支持链式调用
        """
        self.processors.append(processor)
        self.logger.info(f"添加数据处理器: {processor.name}")
        return self

    def set_writer(self, writer: DataWriter) -> "Pipeline":
        """
        设置数据输出器

        Args:
            writer: 数据输出器实例

        Returns:
            管道实例，支持链式调用
        """
        self.writer = writer
        self.logger.info(f"设置数据输出器: {writer.name}")
        return self

    def validate(self) -> bool:
        """
        验证管道配置是否有效

        Returns:
            管道配置是否有效
        """
        if self.loader is None:
            self.logger.error("未设置数据加载器")
            return False

        if self.writer is None:
            self.logger.error("未设置数据输出器")
            return False

        return True

    def run(self) -> bool:
        """
        运行数据处理管道

        Returns:
            处理是否成功
        """
        if not self.validate():
            return False

        self.logger.info(f"开始执行管道: {self.name}")
        start_time = time.time()

        try:
            # 加载数据
            self.logger.info(f"开始加载数据: {self.loader.name}")
            data = self.loader.load()
            if data.empty:
                self.logger.warning("加载的数据为空")
            else:
                self.logger.info(f"成功加载数据: {len(data)} 行")

            # 处理数据
            for i, processor in enumerate(self.processors):
                if data.empty:
                    self.logger.warning(f"跳过处理器 {processor.name}，因为输入数据为空")
                    continue

                self.logger.info(
                    f"开始处理数据: {processor.name} ({i+1}/{len(self.processors)})"
                )
                processor_start_time = time.time()

                data = processor.process(data)

                processor_elapsed_time = time.time() - processor_start_time
                self.logger.info(
                    f"处理器 {processor.name} 完成，耗时: {processor_elapsed_time:.2f}秒"
                )

                if data.empty:
                    self.logger.warning(f"处理器 {processor.name} 处理后，数据为空")

            # 输出数据
            if not data.empty:
                self.logger.info(f"开始输出数据: {self.writer.name}")
                self.writer.write(data)
                self.logger.info(f"数据输出完成: {len(data)} 行")
            else:
                self.logger.warning("没有数据可输出")

            elapsed_time = time.time() - start_time
            self.logger.info(f"管道执行完成，总耗时: {elapsed_time:.2f}秒")
            return True

        except Exception as e:
            self.logger.error(f"管道执行出错: {str(e)}", exc_info=True)
            return False

    @classmethod
    def from_config(cls, config: Dict[str, Any], registry) -> "Pipeline":
        """
        从配置创建管道

        Args:
            config: 管道配置
            registry: 组件注册表

        Returns:
            创建的管道实例
        """
        name = config.get("name", "ConfigPipeline")
        pipeline = cls(name=name)
        logger = logging.getLogger(f"DataPipeline.Pipeline.{name}")

        # 加载器配置
        loader_config = config.get("loader", {})
        loader_name = loader_config.get("name")
        loader_params = loader_config.get("params", {})

        if loader_name:
            try:
                loader = registry.get_loader(loader_name, **loader_params)
                pipeline.set_loader(loader)
            except Exception as e:
                logger.error(f"创建加载器 {loader_name} 失败: {str(e)}")
                raise

        # 处理器配置
        for processor_config in config.get("processors", []):
            processor_name = processor_config.get("name")
            processor_params = processor_config.get("params", {})

            if processor_name:
                try:
                    processor = registry.get_processor(
                        processor_name, **processor_params
                    )
                    pipeline.add_processor(processor)
                except Exception as e:
                    logger.error(f"创建处理器 {processor_name} 失败: {str(e)}")
                    raise

        # 输出器配置
        writer_config = config.get("writer", {})
        writer_name = writer_config.get("name")
        writer_params = writer_config.get("params", {})

        if writer_name:
            try:
                writer = registry.get_writer(writer_name, **writer_params)
                pipeline.set_writer(writer)
            except Exception as e:
                logger.error(f"创建输出器 {writer_name} 失败: {str(e)}")
                raise

        return pipeline

    @classmethod
    def from_json(cls, json_path: str, registry) -> "Pipeline":
        """
        从JSON配置文件创建管道

        Args:
            json_path: JSON配置文件路径
            registry: 组件注册表

        Returns:
            创建的管道实例
        """
        logger = logging.getLogger("DataPipeline.Pipeline")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            logger.info(f"从配置文件加载管道: {json_path}")
            return cls.from_config(config, registry)

        except Exception as e:
            logger.error(f"从配置文件创建管道失败: {str(e)}")
            raise
