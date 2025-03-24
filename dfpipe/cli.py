#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import sys

from dfpipe.core.pipeline import Pipeline
from dfpipe.core.registry import ComponentRegistry
from dfpipe.utils.logging import setup_logging


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="数据流水线处理工具")

    # 基本参数
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--log-file", help="日志文件名")
    parser.add_argument("--log-level", default="INFO", help="日志级别")

    # 简单模式的参数
    parser.add_argument("--input-dir", default="data", help="输入目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--loader", default="CSVLoader", help="数据加载器名称")
    parser.add_argument("--writer", default="CSVWriter", help="数据输出器名称")
    parser.add_argument("--processors", help="数据处理器列表，逗号分隔")

    args = parser.parse_args()

    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # 初始化日志
    logger = setup_logging(log_file=args.log_file, level=log_level)
    logger.info("数据流水线工具启动")

    # 自动发现组件
    ComponentRegistry.auto_discover()

    # 创建并运行管道
    try:
        if args.config:
            # 从配置文件创建管道
            logger.info(f"从配置文件创建管道: {args.config}")

            if not os.path.exists(args.config):
                logger.error(f"配置文件不存在: {args.config}")
                return 1

            pipeline = Pipeline.from_json(args.config, ComponentRegistry)

        else:
            # 使用命令行参数创建简单管道
            logger.info("使用命令行参数创建简单管道")

            # 创建管道
            pipeline = Pipeline(name="CommandLinePipeline")

            # 设置加载器
            try:
                loader = ComponentRegistry.get_loader(
                    args.loader, input_dir=args.input_dir
                )
                pipeline.set_loader(loader)
            except Exception as e:
                logger.error(f"创建加载器失败: {str(e)}")
                return 1

            # 添加处理器
            if args.processors:
                processor_names = [p.strip() for p in args.processors.split(",")]
                for processor_name in processor_names:
                    try:
                        processor = ComponentRegistry.get_processor(processor_name)
                        pipeline.add_processor(processor)
                    except Exception as e:
                        logger.error(f"创建处理器 {processor_name} 失败: {str(e)}")

            # 设置输出器
            try:
                writer = ComponentRegistry.get_writer(
                    args.writer, output_dir=args.output_dir
                )
                pipeline.set_writer(writer)
            except Exception as e:
                logger.error(f"创建输出器失败: {str(e)}")
                return 1

        # 验证管道
        if not pipeline.validate():
            logger.error("管道配置无效")
            return 1

        # 运行管道
        logger.info(f"开始运行管道: {pipeline.name}")
        success = pipeline.run()

        if success:
            logger.info("管道运行成功")
            return 0
        else:
            logger.error("管道运行失败")
            return 1

    except Exception as e:
        logger.error(f"运行管道时出错: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
