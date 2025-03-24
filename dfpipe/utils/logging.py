#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    设置日志系统

    Args:
        log_dir: 日志目录
        log_file: 日志文件名，如果不指定则自动生成
        level: 日志等级
        console: 是否输出到控制台

    Returns:
        根日志对象
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 如果未指定日志文件名，使用当前时间生成
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"pipeline_{timestamp}.log"

    log_path = os.path.join(log_dir, log_file)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 如果需要，创建控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    logger = logging.getLogger("DataPipeline")
    logger.info(f"日志系统初始化完成，日志文件: {log_path}")

    return logger
