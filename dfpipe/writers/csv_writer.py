#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from dfpipe.core.base import DataWriter
from dfpipe.core.registry import ComponentRegistry


@ComponentRegistry.register_writer
class CSVWriter(DataWriter):
    """将数据输出到CSV文件"""

    def __init__(
        self,
        output_dir: str = "output",
        filename: str = None,
        use_timestamp: bool = True,
        encoding: str = "utf-8",
        index: bool = False,
        **kwargs,
    ):
        """
        初始化CSV数据输出器

        Args:
            output_dir: 输出目录
            filename: 输出文件名，如果不指定则自动生成
            use_timestamp: 是否在文件名中使用时间戳
            encoding: 文件编码
            index: 是否包含索引
            **kwargs: 其他传递给pandas to_csv的参数
        """
        super().__init__(name="CSVWriter", description=f"将数据输出到{output_dir}目录的CSV文件")
        self.output_dir = output_dir
        self.filename = filename
        self.use_timestamp = use_timestamp
        self.encoding = encoding
        self.index = index
        self.csv_options = kwargs

    def write(self, data: pd.DataFrame) -> None:
        """
        将数据输出到CSV文件

        Args:
            data: 要输出的数据框
        """
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"创建输出目录: {self.output_dir}")

        # 生成输出文件名
        if self.filename:
            filename = self.filename
            # 如果需要，添加时间戳
            if self.use_timestamp:
                name, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}{ext}"
        else:
            # 自动生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.csv"

        # 完整的输出路径
        output_path = os.path.join(self.output_dir, filename)

        try:
            # 保存数据到CSV
            data.to_csv(
                output_path,
                encoding=self.encoding,
                index=self.index,
                **self.csv_options,
            )
            self.logger.info(f"成功保存 {len(data)} 行数据到 {output_path}")

        except Exception as e:
            self.logger.error(f"保存数据到 {output_path} 失败: {str(e)}")
            raise
