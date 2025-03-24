#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
from typing import List, Optional

import pandas as pd

from dfpipe.core.base import DataLoader
from dfpipe.core.registry import ComponentRegistry


@ComponentRegistry.register_loader
class CSVLoader(DataLoader):
    """从CSV文件加载数据"""

    def __init__(
        self,
        input_dir: str = "data",
        file_pattern: str = "*.csv",
        encoding: str = "utf-8",
        **kwargs,
    ):
        """
        初始化CSV数据加载器

        Args:
            input_dir: 输入目录
            file_pattern: 文件模式
            encoding: 文件编码
            **kwargs: 其他参数
        """
        super().__init__(name="CSVLoader", description=f"从{input_dir}目录加载CSV格式的数据")
        self.input_dir = input_dir
        self.file_pattern = file_pattern
        self.encoding = encoding
        self.csv_options = kwargs  # 传递给pandas的其他CSV读取选项

    def load(self) -> pd.DataFrame:
        """
        从CSV文件加载数据

        Returns:
            加载的数据框
        """
        self.logger.info(f"开始从{self.input_dir}加载CSV数据")

        # 确保输入目录存在
        if not os.path.exists(self.input_dir):
            self.logger.warning(f"输入目录不存在: {self.input_dir}")
            os.makedirs(self.input_dir)
            return pd.DataFrame()

        # 查找匹配的CSV文件
        pattern = os.path.join(self.input_dir, self.file_pattern)
        files = glob.glob(pattern)
        self.logger.info(f"找到 {len(files)} 个CSV文件")

        if not files:
            self.logger.warning(
                f"在 {self.input_dir} 目录中未找到匹配的CSV文件: {self.file_pattern}"
            )
            return pd.DataFrame()  # 返回空DataFrame

        # 读取所有CSV文件并合并
        result_dfs = []
        for file_path in files:
            try:
                # 加载CSV文件
                filename = os.path.basename(file_path)
                self.logger.info(f"正在加载文件: {filename}")

                # 读取CSV文件
                df = pd.read_csv(file_path, encoding=self.encoding, **self.csv_options)

                # 添加源文件列
                df["source_file"] = filename

                # 记录行数
                self.logger.info(f"从 {filename} 加载了 {len(df)} 行数据")

                # 添加到结果列表
                result_dfs.append(df)

            except Exception as e:
                self.logger.error(f"加载文件 {file_path} 失败: {str(e)}")

        # 合并所有数据框
        if not result_dfs:
            self.logger.warning("没有成功加载任何CSV文件")
            return pd.DataFrame()

        # 合并所有数据框
        result = pd.concat(result_dfs, ignore_index=True)
        self.logger.info(f"总共加载了 {len(result_dfs)} 个文件，{len(result)} 行数据")

        return result
