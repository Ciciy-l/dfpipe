#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用代码API构建和运行数据处理管道的示例
"""

import os
import sys

import pandas as pd

# 将包所在目录添加到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dfpipe import ComponentRegistry, Pipeline, setup_logging
from dfpipe.utils.logging import setup_logging


def main():
    # 设置日志
    logger = setup_logging()
    logger.info("示例脚本启动")

    # 自动发现组件
    ComponentRegistry.auto_discover()

    # 创建一些示例数据（在实际使用中，数据会由加载器加载）
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 17, 30, 16, 22],
            "score": [85, 92, 78, 88, 95],
        }
    )
    df.to_csv("data/example.csv", index=False)
    logger.info("创建了示例数据文件: data/example.csv")

    # 创建管道
    pipeline = Pipeline(name="示例管道")

    # 设置加载器
    loader = ComponentRegistry.get_loader("CSVLoader", input_dir="data")
    pipeline.set_loader(loader)

    # 添加过滤处理器（只保留年龄>=18的记录）
    def is_adult(age):
        return age >= 18

    filter_processor = ComponentRegistry.get_processor(
        "FilterProcessor", column="age", condition=is_adult
    )
    pipeline.add_processor(filter_processor)

    # 添加转换处理器（分数转为等级）
    def score_to_grade(score):
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        else:
            return "D"

    transform_processor = ComponentRegistry.get_processor(
        "TransformProcessor",
        column="score",
        transform_func=score_to_grade,
        target_column="grade",
    )
    pipeline.add_processor(transform_processor)

    # 添加列操作处理器（添加一个新列）
    column_processor = ComponentRegistry.get_processor(
        "ColumnProcessor", operation="add", column="status", value="active"
    )
    pipeline.add_processor(column_processor)

    # 设置输出器
    writer = ComponentRegistry.get_writer(
        "CSVWriter", output_dir="output", filename="processed_example.csv"
    )
    pipeline.set_writer(writer)

    # 运行管道
    logger.info("开始运行示例管道")
    pipeline.run()

    logger.info("示例脚本执行完成")


if __name__ == "__main__":
    main()
