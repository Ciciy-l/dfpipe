#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速开始示例

展示如何导入和使用dfpipe包
"""

import pandas as pd

from dfpipe import (
    CSVLoader,
    CSVWriter,
    FilterProcessor,
    Pipeline,
    Registry,
    setup_logging,
)


def main():
    # 设置日志
    logger = setup_logging()
    logger.info("快速开始示例启动")

    # 创建示例数据
    df = pd.DataFrame(
        {
            "name": ["张三", "李四", "王五", "赵六", "钱七"],
            "age": [25, 17, 30, 16, 22],
            "city": ["北京", "上海", "广州", "深圳", "杭州"],
        }
    )

    # 保存示例数据
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/people.csv", index=False, encoding="utf-8")
    logger.info("创建了示例数据: data/people.csv")

    # 创建管道
    pipeline = Pipeline(name="快速开始示例")

    # 设置加载器
    loader = CSVLoader(input_dir="data", encoding="utf-8")
    pipeline.set_loader(loader)

    # 添加过滤器 - 仅保留成年人
    def is_adult(age):
        return age >= 18

    filter_processor = FilterProcessor(column="age", condition=is_adult)
    pipeline.add_processor(filter_processor)

    # 设置输出器
    writer = CSVWriter(output_dir="output", filename="adults.csv")
    pipeline.set_writer(writer)

    # 运行管道
    logger.info("开始执行管道")
    success = pipeline.run()

    if success:
        logger.info("管道执行成功")
    else:
        logger.error("管道执行失败")

    logger.info("示例结束")


if __name__ == "__main__":
    import os

    main()
