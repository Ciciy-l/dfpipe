#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, call, patch

from dfpipe.utils.logging import setup_logging


class TestLogging(unittest.TestCase):
    """测试日志模块"""

    def setUp(self):
        """设置测试环境"""
        # 创建临时目录用于测试日志文件
        self.temp_dir = tempfile.mkdtemp()

        # 保存和清除现有的 root handlers，以便在测试后恢复
        self.original_handlers = logging.getLogger().handlers.copy()
        logging.getLogger().handlers.clear()

        # 保存原始日志级别
        self.original_level = logging.getLogger().level

    def tearDown(self):
        """清理测试环境"""
        # 清除临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # 恢复原始的 root handlers
        logging.getLogger().handlers.clear()
        for handler in self.original_handlers:
            logging.getLogger().addHandler(handler)

        # 恢复原始日志级别
        logging.getLogger().setLevel(self.original_level)

    def test_setup_logging_default_parameters(self):
        """测试使用默认参数设置日志"""
        with patch("dfpipe.utils.logging.datetime") as mock_datetime:
            # 设置模拟的当前时间
            mock_now = MagicMock()
            mock_now.strftime.return_value = "20230101_120000"
            mock_datetime.now.return_value = mock_now

            with patch("os.path.exists", return_value=False), patch(
                "os.makedirs"
            ) as mock_makedirs, patch(
                "logging.FileHandler"
            ) as mock_file_handler, patch(
                "logging.StreamHandler"
            ) as mock_stream_handler:
                # 创建模拟的处理器实例并设置level属性
                mock_file_handler_instance = MagicMock()
                mock_file_handler_instance.level = logging.INFO
                mock_file_handler.return_value = mock_file_handler_instance

                mock_stream_handler_instance = MagicMock()
                mock_stream_handler_instance.level = logging.INFO
                mock_stream_handler.return_value = mock_stream_handler_instance

                # 调用setup_logging
                logger = setup_logging()

                # 验证日志目录创建
                mock_makedirs.assert_called_once_with("logs")

                # 验证生成的日志文件名
                expected_log_file = "pipeline_20230101_120000.log"
                expected_log_path = os.path.join("logs", expected_log_file)
                mock_file_handler.assert_called_once()
                file_handler_args = mock_file_handler.call_args[0]
                self.assertEqual(file_handler_args[0], expected_log_path)

                # 验证创建了控制台处理器
                mock_stream_handler.assert_called_once_with(sys.stdout)

                # 验证返回的logger
                self.assertIsInstance(logger, logging.Logger)
                self.assertEqual(logger.name, "DataPipeline")

    def test_setup_logging_custom_parameters(self):
        """测试使用自定义参数设置日志"""
        custom_log_dir = os.path.join(self.temp_dir, "custom_logs")
        custom_log_file = "custom.log"
        custom_level = logging.DEBUG

        with patch("os.path.exists", return_value=False), patch(
            "os.makedirs"
        ) as mock_makedirs, patch("logging.FileHandler") as mock_file_handler, patch(
            "logging.StreamHandler"
        ) as mock_stream_handler:
            # 创建模拟的处理器实例并设置level属性
            mock_file_handler_instance = MagicMock()
            mock_file_handler_instance.level = logging.DEBUG
            mock_file_handler.return_value = mock_file_handler_instance

            mock_stream_handler_instance = MagicMock()
            mock_stream_handler_instance.level = logging.DEBUG
            mock_stream_handler.return_value = mock_stream_handler_instance

            # 调用setup_logging
            logger = setup_logging(
                log_dir=custom_log_dir,
                log_file=custom_log_file,
                level=custom_level,
                console=True,
            )

            # 验证日志目录创建
            mock_makedirs.assert_called_once_with(custom_log_dir)

            # 验证日志文件
            expected_log_path = os.path.join(custom_log_dir, custom_log_file)
            mock_file_handler.assert_called_once()
            file_handler_args = mock_file_handler.call_args[0]
            self.assertEqual(file_handler_args[0], expected_log_path)

            # 验证日志级别
            self.assertEqual(logging.getLogger().level, custom_level)

            # 验证创建了控制台处理器
            mock_stream_handler.assert_called_once()

    def test_setup_logging_existing_directory(self):
        """测试使用已存在的日志目录设置日志"""
        with patch("os.path.exists", return_value=True), patch(
            "os.makedirs"
        ) as mock_makedirs, patch("logging.FileHandler") as mock_file_handler, patch(
            "logging.StreamHandler"
        ) as mock_stream_handler:
            # 创建模拟的处理器实例并设置level属性
            mock_file_handler_instance = MagicMock()
            mock_file_handler_instance.level = logging.INFO
            mock_file_handler.return_value = mock_file_handler_instance

            mock_stream_handler_instance = MagicMock()
            mock_stream_handler_instance.level = logging.INFO
            mock_stream_handler.return_value = mock_stream_handler_instance

            # 调用setup_logging
            setup_logging()

            # 验证没有创建日志目录
            mock_makedirs.assert_not_called()

    def test_setup_logging_no_console(self):
        """测试不使用控制台输出设置日志"""
        with patch("os.path.exists", return_value=True), patch(
            "logging.FileHandler"
        ) as mock_file_handler, patch("logging.StreamHandler") as mock_stream_handler:
            # 创建模拟的处理器实例并设置level属性
            mock_file_handler_instance = MagicMock()
            mock_file_handler_instance.level = logging.INFO
            mock_file_handler.return_value = mock_file_handler_instance

            # 调用setup_logging，不使用控制台输出
            setup_logging(console=False)

            # 验证没有创建控制台处理器
            mock_stream_handler.assert_not_called()

    def test_setup_logging_clear_existing_handlers(self):
        """测试清除现有处理器"""
        # 添加一个测试处理器
        test_handler = logging.NullHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(test_handler)

        with patch("os.path.exists", return_value=True), patch(
            "logging.FileHandler"
        ) as mock_file_handler, patch("logging.StreamHandler") as mock_stream_handler:
            # 创建模拟的处理器实例并设置level属性
            mock_file_handler_instance = MagicMock()
            mock_file_handler_instance.level = logging.INFO
            mock_file_handler.return_value = mock_file_handler_instance

            mock_stream_handler_instance = MagicMock()
            mock_stream_handler_instance.level = logging.INFO
            mock_stream_handler.return_value = mock_stream_handler_instance

            # 调用setup_logging
            setup_logging()

            # 验证测试处理器已被移除
            self.assertNotIn(test_handler, root_logger.handlers)

    def test_setup_logging_real_file(self):
        """使用真实文件系统测试日志设置"""
        # 创建自定义日志目录和文件名
        custom_log_dir = os.path.join(self.temp_dir, "real_logs")
        custom_log_file = "real_test.log"

        # 调用setup_logging（不使用模拟，测试实际文件系统）
        logger = setup_logging(
            log_dir=custom_log_dir,
            log_file=custom_log_file,
            level=logging.DEBUG,
            console=False,  # 关闭控制台输出以便于测试
        )

        # 验证日志目录已创建
        self.assertTrue(os.path.exists(custom_log_dir))

        # 验证日志文件已创建
        log_path = os.path.join(custom_log_dir, custom_log_file)
        self.assertTrue(os.path.exists(log_path))

        # 写入测试日志消息
        test_message = "这是一条测试日志消息"
        logger.debug(test_message)

        # 验证消息已写入文件
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)

    def test_setup_logging_formatters(self):
        """测试日志格式化器设置"""
        with patch("os.path.exists", return_value=True), patch(
            "logging.FileHandler"
        ) as mock_file_handler, patch("logging.StreamHandler") as mock_stream_handler:
            # 创建模拟的处理器实例
            mock_file_handler_instance = MagicMock()
            mock_file_handler_instance.level = logging.INFO
            mock_file_handler.return_value = mock_file_handler_instance

            mock_stream_handler_instance = MagicMock()
            mock_stream_handler_instance.level = logging.INFO
            mock_stream_handler.return_value = mock_stream_handler_instance

            # 调用setup_logging
            setup_logging()

            # 验证文件处理器的格式化器
            mock_file_handler_instance.setFormatter.assert_called_once()
            file_formatter = mock_file_handler_instance.setFormatter.call_args[0][0]
            self.assertEqual(
                file_formatter._fmt,
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            # 验证控制台处理器的格式化器
            mock_stream_handler_instance.setFormatter.assert_called_once()
            console_formatter = mock_stream_handler_instance.setFormatter.call_args[0][
                0
            ]
            self.assertEqual(
                console_formatter._fmt, "%(asctime)s - %(levelname)s - %(message)s"
            )


if __name__ == "__main__":
    unittest.main()
