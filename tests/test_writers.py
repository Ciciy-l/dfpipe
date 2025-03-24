#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据输出器测试模块
包含所有dfpipe数据输出器组件的单元测试
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from dfpipe.core.base import DataWriter
from dfpipe.writers.csv_writer import CSVWriter
from tests import create_test_data


class BaseWriterTest(unittest.TestCase):
    """所有输出器测试的基类"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试数据 - 注意这里没有自动添加id列
        self.test_data = create_test_data(rows=10, columns=3)
        self.test_data.columns = ["id", "value", "category"]

        # 创建临时目录用于测试文件
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        # 删除测试文件和目录
        shutil.rmtree(self.temp_dir)

    def validate_written_data(self, file_path, expected_data=None, **read_options):
        """验证写入的数据是否符合预期"""
        expected_data = expected_data or self.test_data

        # 基本文件验证
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(os.path.getsize(file_path) > 0)


class TestCSVWriter(BaseWriterTest):
    """测试CSV文件输出器"""

    def setUp(self):
        """设置测试环境"""
        super().setUp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def test_write_csv(self):
        """测试写入CSV文件"""
        # 创建CSV输出器
        writer = CSVWriter(
            output_dir=self.output_dir, filename="test_output.csv", use_timestamp=False
        )

        # 写入数据
        writer.write(self.test_data)

        # 检查输出文件
        output_path = os.path.join(self.output_dir, "test_output.csv")
        self.validate_written_data(output_path)

        # 读取并验证内容
        read_data = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(
            read_data, self.test_data, check_dtype=False  # CSV写入可能改变数据类型
        )

    def test_write_with_timestamp(self):
        """测试使用时间戳写入CSV"""
        # 创建带有时间戳的CSV输出器
        writer = CSVWriter(
            output_dir=self.output_dir, filename="test_output.csv", use_timestamp=True
        )

        # 写入数据
        writer.write(self.test_data)

        # 检查输出目录中的文件
        files = os.listdir(self.output_dir)
        self.assertTrue(len(files) > 0)

        # 验证文件名包含时间戳
        csv_files = [
            f for f in files if f.startswith("test_output_") and f.endswith(".csv")
        ]
        self.assertEqual(len(csv_files), 1)

        # 读取并验证内容
        output_path = os.path.join(self.output_dir, csv_files[0])
        read_data = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(read_data, self.test_data, check_dtype=False)

    def test_write_auto_filename(self):
        """测试自动生成文件名"""
        # 创建不指定文件名的CSV输出器
        writer = CSVWriter(output_dir=self.output_dir)

        # 写入数据
        writer.write(self.test_data)

        # 检查输出目录中的文件
        files = os.listdir(self.output_dir)
        self.assertTrue(len(files) > 0)

        # 验证文件名格式
        csv_files = [f for f in files if f.startswith("data_") and f.endswith(".csv")]
        self.assertEqual(len(csv_files), 1)

        # 读取并验证内容
        output_path = os.path.join(self.output_dir, csv_files[0])
        read_data = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(read_data, self.test_data, check_dtype=False)

    def test_write_with_options(self):
        """测试使用不同选项写入CSV"""
        # 创建带选项的CSV输出器
        writer = CSVWriter(
            output_dir=self.output_dir,
            filename="test_options.csv",
            use_timestamp=False,
            index=True,
            sep="|",
        )

        # 写入数据
        writer.write(self.test_data)

        # 验证文件
        output_path = os.path.join(self.output_dir, "test_options.csv")
        self.validate_written_data(output_path)

        # 读取并验证内容
        read_data = pd.read_csv(output_path, sep="|", index_col=0)

        # 重置索引以便比较（因为写入时添加了索引列）
        test_data_with_index = self.test_data.copy()
        pd.testing.assert_frame_equal(
            read_data, test_data_with_index, check_dtype=False
        )

    def test_write_nonexistent_directory(self):
        """测试写入不存在的目录"""
        # 创建指向不存在目录的输出器
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        writer = CSVWriter(
            output_dir=nonexistent_dir,
            filename="test_create_dir.csv",
            use_timestamp=False,
        )

        # 写入数据 - 应该自动创建目录
        writer.write(self.test_data)

        # 验证目录和文件是否被创建
        self.assertTrue(os.path.exists(nonexistent_dir))
        output_path = os.path.join(nonexistent_dir, "test_create_dir.csv")
        self.assertTrue(os.path.exists(output_path))

        # 读取并验证内容
        read_data = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(read_data, self.test_data, check_dtype=False)

    def test_write_error_handling(self):
        """测试写入错误处理"""
        # 创建CSV输出器
        writer = CSVWriter(
            output_dir=self.output_dir, filename="test_error.csv", use_timestamp=False
        )

        # 模拟to_csv抛出异常
        with patch("pandas.DataFrame.to_csv", side_effect=Exception("模拟写入错误")):
            # 验证异常被抛出
            with self.assertRaises(Exception):
                writer.write(self.test_data)


@pytest.mark.usefixtures("mock_csv_writer")
class TestCSVWriterWithMock(unittest.TestCase):
    """使用mock测试CSVWriter"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试数据 - 注意这里没有自动添加id列
        self.test_data = create_test_data(rows=10, columns=3)
        self.test_data.columns = ["id", "value", "category"]

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_write_csv_with_mock(self, mock_to_csv, mock_makedirs, mock_exists):
        """使用mock测试CSV写入"""
        # 设置模拟行为
        mock_exists.return_value = True

        # 创建CSV输出器并写入数据
        writer = CSVWriter(
            output_dir="fake_dir", filename="test_output.csv", use_timestamp=False
        )
        writer.write(self.test_data)

        # 验证调用
        mock_exists.assert_called_once_with("fake_dir")
        mock_makedirs.assert_not_called()  # 目录已存在，不应该创建
        mock_to_csv.assert_called_once()

        # 验证to_csv的参数
        args, kwargs = mock_to_csv.call_args
        self.assertEqual(kwargs.get("index"), False)
        self.assertEqual(kwargs.get("encoding"), "utf-8")

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_create_nonexistent_directory_with_mock(
        self, mock_to_csv, mock_makedirs, mock_exists
    ):
        """使用mock测试创建不存在的目录"""
        # 设置模拟行为
        mock_exists.return_value = False

        # 创建CSV输出器并写入数据
        writer = CSVWriter(
            output_dir="nonexistent_dir",
            filename="test_output.csv",
            use_timestamp=False,
        )
        writer.write(self.test_data)

        # 验证调用
        mock_exists.assert_called_once_with("nonexistent_dir")
        mock_makedirs.assert_called_once_with("nonexistent_dir")
        mock_to_csv.assert_called_once()

    @patch("os.path.exists")
    @patch("pandas.DataFrame.to_csv")
    def test_auto_filename_with_mock(self, mock_to_csv, mock_exists):
        """使用mock测试自动生成文件名"""
        # 设置模拟行为
        mock_exists.return_value = True

        # 使用固定的日期时间值
        timestamp = "20230101_120000"

        # 直接模拟csv_writer.py中的datetime.now函数
        datetime_patcher = patch("dfpipe.writers.csv_writer.datetime")
        mock_datetime = datetime_patcher.start()
        try:
            # 重要：必须模拟now和strftime
            mock_now = MagicMock()
            mock_now.strftime.return_value = timestamp
            mock_datetime.now.return_value = mock_now

            # 创建CSV输出器并写入数据
            writer = CSVWriter(output_dir="fake_dir", filename=None)  # 自动生成文件名
            writer.write(self.test_data)

            # 验证datetime.now被调用
            mock_datetime.now.assert_called_once()

            # 验证使用了正确的自动生成文件名
            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            expected_path = os.path.join("fake_dir", f"data_{timestamp}.csv")
            self.assertEqual(args[0], expected_path)
        finally:
            datetime_patcher.stop()

    def test_timestamp_filename_with_mock(self):
        """使用mock测试带时间戳的文件名"""
        # 设置模拟行为
        with patch("os.path.exists", return_value=True) as mock_exists, patch(
            "pandas.DataFrame.to_csv"
        ) as mock_to_csv, patch("dfpipe.writers.csv_writer.datetime") as mock_datetime:
            # 重要: 必须正确设置mock_datetime.now().strftime()的行为
            mock_now = MagicMock()
            mock_now.strftime.return_value = "20230101_120000"
            mock_datetime.now.return_value = mock_now

            # 创建CSV输出器并写入数据
            writer = CSVWriter(
                output_dir="fake_dir", filename="test_output.csv", use_timestamp=True
            )
            writer.write(self.test_data)

            # 验证调用
            mock_datetime.now.assert_called_once()
            mock_now.strftime.assert_called_once_with("%Y%m%d_%H%M%S")

            # 验证使用了正确的文件名
            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            expected_path = os.path.join("fake_dir", "test_output_20230101_120000.csv")
            self.assertEqual(args[0], expected_path)

    @patch("os.path.exists")
    @patch("pandas.DataFrame.to_csv")
    @patch("dfpipe.writers.csv_writer.datetime")
    def test_write_with_mock_fixture_style(
        self, mock_datetime, mock_to_csv, mock_exists
    ):
        """使用mock测试CSV写入（仿fixture风格）"""
        # 设置模拟行为
        mock_exists.return_value = True

        # 设置固定的时间戳
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20230101_120000"
        mock_datetime.now.return_value = mock_now

        # 准备测试数据
        test_data = pd.DataFrame(
            {"id": [1, 2, 3], "value": [10, 20, 30], "category": ["A", "B", "C"]}
        )

        # 创建写入器并写入数据
        writer = CSVWriter(
            output_dir="/fake/dir", filename="test.csv", use_timestamp=False
        )
        writer.write(test_data)

        # 验证调用
        mock_exists.assert_any_call("/fake/dir")
        mock_to_csv.assert_called_once()

        # 验证写入的文件路径和其他参数
        args, kwargs = mock_to_csv.call_args
        expected_path = os.path.join("/fake/dir", "test.csv")
        self.assertEqual(args[0], expected_path)

        # 验证index参数（默认为False）
        self.assertFalse(kwargs.get("index", True))

    @patch("os.path.exists")
    @patch("pandas.DataFrame.to_csv")
    def test_csv_options_with_mock(self, mock_to_csv, mock_exists):
        """使用mock测试CSV写入选项"""
        # 设置模拟行为
        mock_exists.return_value = True

        # 创建带有自定义选项的CSV输出器
        writer = CSVWriter(
            output_dir="fake_dir",
            filename="test_options.csv",
            use_timestamp=False,
            index=True,
            sep="|",
            quoting=1,  # 额外CSV选项
            quotechar='"',  # 额外CSV选项
        )
        writer.write(self.test_data)

        # 验证to_csv被调用的参数
        args, kwargs = mock_to_csv.call_args
        self.assertEqual(kwargs.get("index"), True)
        self.assertEqual(kwargs.get("sep"), "|")
        self.assertEqual(kwargs.get("quoting"), 1)
        self.assertEqual(kwargs.get("quotechar"), '"')


@pytest.mark.usefixtures("test_df")
class TestCSVWriterPytest:
    """使用pytest测试CSVWriter"""

    def test_write_with_timestamp(self, test_df, temp_dir):
        """测试带时间戳的写入"""
        # 准备固定的时间戳并模拟datetime
        timestamp = "20230101_120000"

        with patch("dfpipe.writers.csv_writer.datetime") as mock_datetime:
            # 设置mock_datetime的行为
            mock_now = MagicMock()
            mock_now.strftime.return_value = timestamp
            mock_datetime.now.return_value = mock_now

            # 创建输出目录
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # 创建写入器
            writer = CSVWriter(
                output_dir=output_dir, filename="test_output.csv", use_timestamp=True
            )

            # 写入数据
            writer.write(test_df)

            # 验证文件是否已创建
            expected_path = os.path.join(output_dir, f"test_output_{timestamp}.csv")
            assert os.path.exists(expected_path)

            # 验证文件内容
            saved_data = pd.read_csv(expected_path)
            assert len(saved_data) == len(test_df)
            assert set(saved_data.columns) == set(test_df.columns)

    def test_with_mock_fixtures(self, test_df):
        """使用mock测试CSV写入器"""
        # 设置mock
        with patch("os.path.exists", return_value=True) as mock_exists, patch(
            "pandas.DataFrame.to_csv"
        ) as mock_to_csv, patch("dfpipe.writers.csv_writer.datetime") as mock_datetime:
            # 设置固定的时间戳
            timestamp = "20230101_120000"
            mock_now = MagicMock()
            mock_now.strftime.return_value = timestamp
            mock_datetime.now.return_value = mock_now

            # 创建输出器并写入数据
            writer = CSVWriter(
                output_dir="fixture_dir",
                filename="test_mock_fixture.csv",
                use_timestamp=True,
            )
            writer.write(test_df)

            # 验证调用和结果
            mock_exists.assert_called_with("fixture_dir")
            mock_datetime.now.assert_called_once()
            mock_to_csv.assert_called_once()

            # 验证使用了正确的文件名（带时间戳）
            args, kwargs = mock_to_csv.call_args
            expected_path = os.path.join(
                "fixture_dir", f"test_mock_fixture_{timestamp}.csv"
            )
            assert args[0] == expected_path


class TestCustomWriter(unittest.TestCase):
    """测试自定义数据输出器"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.test_data = create_test_data(rows=5, columns=2)
        self.test_data.columns = ["id", "value"]

        # 定义自定义输出器
        class CustomWriter(DataWriter):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.written_data = None

            def write(self, data):
                self.written_data = data.copy()
                # 不返回任何值，符合DataWriter接口

        self.writer_class = CustomWriter

    def test_custom_writer(self):
        """测试自定义输出器"""
        # 创建自定义输出器实例
        writer = self.writer_class(name="custom_writer", description="自定义数据输出器")

        # 写入数据
        writer.write(self.test_data)

        # 验证结果
        self.assertIsInstance(writer.written_data, pd.DataFrame)
        pd.testing.assert_frame_equal(writer.written_data, self.test_data)


if __name__ == "__main__":
    unittest.main()
