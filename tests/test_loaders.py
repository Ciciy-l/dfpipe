#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载器测试模块
包含所有dfpipe数据加载器组件的单元测试
"""

import unittest
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import glob
from unittest.mock import patch, MagicMock, call

from dfpipe.core.base import DataLoader
# 假设这些是实际的加载器实现，需要根据实际情况导入
# from dfpipe.loaders.file_loader import CsvLoader, ExcelLoader, JsonLoader
# from dfpipe.loaders.db_loader import SqlLoader
from dfpipe.loaders.csv_loader import CSVLoader
from tests import create_test_data


class BaseLoaderTest(unittest.TestCase):
    """所有加载器测试的基类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录用于测试文件
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        os.makedirs(self.input_dir, exist_ok=True)
        
        # 创建测试数据 - 注意这里没有自动添加id列
        self.test_data1 = create_test_data(rows=5, columns=3)
        self.test_data1.columns = ["id", "value", "category"]
        
        self.test_data2 = create_test_data(rows=7, columns=3)
        self.test_data2.columns = ["id", "value", "category"]
        
        # 写入CSV测试文件
        self.test_data1.to_csv(os.path.join(self.input_dir, "data1.csv"), index=False)
        self.test_data2.to_csv(os.path.join(self.input_dir, "data2.csv"), index=False)
        
    def tearDown(self):
        """清理测试环境"""
        # 删除测试文件和目录
        shutil.rmtree(self.temp_dir)
        
    def validate_loaded_data(self, result_data, expected_data=None):
        """验证加载的数据是否符合预期"""
        # 验证结果是否为DataFrame
        self.assertIsInstance(result_data, pd.DataFrame)
        
        # 验证结果不为空
        self.assertFalse(result_data.empty)
        
        # 验证数据是否包含预期的列
        expected_columns = ["id", "value", "category"]
        for col in expected_columns:
            self.assertIn(col, result_data.columns)
            
        # 验证是否包含source_file列
        self.assertIn("source_file", result_data.columns)
        
        # 如果提供了预期数据，验证行数是否一致
        if expected_data is not None:
            self.assertEqual(len(result_data), len(expected_data))
            
            # 排除source_file列后比较内容
            result_subset = result_data.drop(columns=["source_file"])
            
            # 如果expected_data和result_subset列名相同，则直接比较
            if set(expected_data.columns) == set(result_subset.columns):
                pd.testing.assert_frame_equal(
                    result_subset.sort_values("id").reset_index(drop=True),
                    expected_data.sort_values("id").reset_index(drop=True)
                )


class TestCSVLoader(BaseLoaderTest):
    """测试CSV文件加载器"""
    
    def test_load_csv(self):
        """测试从CSV文件加载数据"""
        # 确保测试数据目录存在且包含测试文件
        input_dir = os.path.join(self.temp_dir, "csv_files")
        os.makedirs(input_dir, exist_ok=True)
        
        # 创建测试数据
        test_data1 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "category": ["A", "B", "C"]
        })
        test_data2 = pd.DataFrame({
            "id": [4, 5, 6],
            "value": [40, 50, 60],
            "category": ["D", "E", "F"]
        })
        
        # 创建测试文件
        test_data1.to_csv(os.path.join(input_dir, "test1.csv"), index=False)
        test_data2.to_csv(os.path.join(input_dir, "test2.csv"), index=False)
        
        # 创建CSVLoader加载数据
        loader = CSVLoader(input_dir=input_dir, file_pattern="*.csv")
        result = loader.load()
        
        # 验证结果
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 6)  # 两个文件共6行
        self.assertIn("id", result.columns)
        self.assertIn("value", result.columns)
        self.assertIn("category", result.columns)
        self.assertIn("source_file", result.columns)
        
    def test_load_specific_pattern(self):
        """测试使用特定模式加载CSV文件"""
        # 确保测试数据目录存在且包含测试文件
        input_dir = os.path.join(self.temp_dir, "csv_files")
        os.makedirs(input_dir, exist_ok=True)
        
        # 创建测试数据
        test_data1 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "category": ["A", "B", "C"]
        })
        test_data2 = pd.DataFrame({
            "id": [4, 5, 6],
            "value": [40, 50, 60],
            "category": ["D", "E", "F"]
        })
        
        # 创建测试文件
        test_data1.to_csv(os.path.join(input_dir, "test1.csv"), index=False)
        test_data2.to_csv(os.path.join(input_dir, "test2.csv"), index=False)
        
        # 创建CSVLoader加载数据，只加载test1.csv
        loader = CSVLoader(input_dir=input_dir, file_pattern="test1.csv")
        result = loader.load()
        
        # 验证结果
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)  # 只有一个文件的3行
        self.assertIn("id", result.columns)
        self.assertIn("value", result.columns)
        self.assertIn("category", result.columns)
        self.assertIn("source_file", result.columns)
        self.assertEqual(result["source_file"].unique()[0], "test1.csv")

    def test_load_nonexistent_directory(self):
        """测试从不存在的目录加载数据"""
        # 创建指向不存在目录的加载器
        loader = CSVLoader(
            input_dir="/nonexistent/dir",
            file_pattern="*.csv"
        )
        
        # 加载数据 - 应返回空DataFrame
        result = loader.load()
        
        # 验证结果
        self.assertTrue(result.empty)
        self.assertIsInstance(result, pd.DataFrame)

    def test_load_empty_directory(self):
        """测试从空目录加载数据"""
        # 创建空目录
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        # 创建加载器
        loader = CSVLoader(
            input_dir=empty_dir,
            file_pattern="*.csv"
        )
        
        # 加载数据 - 应返回空DataFrame
        result = loader.load()
        
        # 验证结果
        self.assertTrue(result.empty)
        self.assertIsInstance(result, pd.DataFrame)

    def test_load_with_encoding(self):
        """测试使用指定编码加载CSV数据"""
        # 创建测试目录
        input_dir = os.path.join(self.temp_dir, "encoding_test")
        os.makedirs(input_dir, exist_ok=True)
        
        # 创建测试数据
        test_data = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "category": ["测试1", "测试2", "测试3"]  # 包含中文字符
        })
        
        # 创建UTF-8编码的CSV文件
        test_data.to_csv(os.path.join(input_dir, "utf8.csv"), index=False, encoding="utf-8")
        
        # 使用UTF-8编码加载
        loader = CSVLoader(
            input_dir=input_dir,
            file_pattern="*.csv",
            encoding="utf-8"
        )
        result = loader.load()
        
        # 验证结果
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertIn("id", result.columns)
        self.assertIn("value", result.columns)
        self.assertIn("category", result.columns)
        self.assertIn("source_file", result.columns)

    def test_load_with_options(self):
        """测试使用额外参数加载CSV数据"""
        # 创建测试目录
        input_dir = os.path.join(self.temp_dir, "options_test")
        os.makedirs(input_dir, exist_ok=True)
        
        # 创建带有自定义分隔符的CSV文件
        with open(os.path.join(input_dir, "semicolon.csv"), "w") as f:
            f.write("id;value;category\n")
            f.write("1;10;A\n")
            f.write("2;20;B\n")
            f.write("3;30;C\n")
        
        # 使用自定义分隔符加载
        loader = CSVLoader(
            input_dir=input_dir,
            file_pattern="*.csv",
            sep=";"  # 自定义分隔符参数
        )
        result = loader.load()
        
        # 验证结果
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertIn("id", result.columns)
        self.assertIn("value", result.columns)
        self.assertIn("category", result.columns)
        self.assertIn("source_file", result.columns)


class TestCSVLoaderWithMock(unittest.TestCase):
    """使用mock测试CSVLoader类"""
    
    def setUp(self):
        """初始化测试数据"""
        self.test_data = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "category": ["A", "B", "C"]
        })
    
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    def test_load_csv_with_mock(self, mock_read_csv, mock_glob, mock_exists):
        """使用mock测试加载CSV文件"""
        # 设置模拟行为
        mock_exists.return_value = True
        mock_glob.return_value = [
            "/fake/path/data1.csv",
            "/fake/path/data2.csv"
        ]
        
        # 为每个文件设置返回的DataFrame
        df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20], "category": ["A", "B"]})
        df2 = pd.DataFrame({"id": [3, 4], "value": [30, 40], "category": ["C", "D"]})
        
        # 添加source_file列，模拟CSV加载器的行为
        df1_with_source = df1.copy()
        df1_with_source["source_file"] = "data1.csv"
        df2_with_source = df2.copy()
        df2_with_source["source_file"] = "data2.csv"
        
        # 设置read_csv的side_effect
        mock_read_csv.side_effect = [df1_with_source, df2_with_source]
        
        # 创建CSV加载器并加载数据
        loader = CSVLoader(
            input_dir="/fake/path",
            file_pattern="*.csv",
            encoding="utf-8"
        )
        result = loader.load()
        
        # 验证调用
        mock_exists.assert_called_with("/fake/path")
        mock_glob.assert_called_once_with(os.path.join("/fake/path", "*.csv"))
        
        # 验证read_csv调用次数
        self.assertEqual(mock_read_csv.call_count, 2)
        
        # 验证结果包含两个DataFrame的合并
        self.assertEqual(len(result), 4)
        self.assertTrue(all(col in result.columns for col in ["id", "value", "category", "source_file"]))
    
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    def test_load_nonexistent_directory_with_mock(self, mock_makedirs, mock_read_csv, mock_glob, mock_exists):
        """使用mock测试加载不存在的目录"""
        # 设置模拟行为 - 目录不存在
        mock_exists.return_value = False
        
        # 创建CSV加载器并加载数据
        loader = CSVLoader(
            input_dir="/nonexistent/dir",
            file_pattern="*.csv"
        )
        result = loader.load()
        
        # 验证调用 - 只验证最初的目录存在检查
        mock_exists.assert_any_call("/nonexistent/dir")
        mock_makedirs.assert_called_once_with("/nonexistent/dir")
        mock_glob.assert_not_called()
        mock_read_csv.assert_not_called()
        
        # 验证结果是空DataFrame
        self.assertTrue(result.empty)
    
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    def test_load_no_matching_files_with_mock(self, mock_read_csv, mock_glob, mock_exists):
        """使用mock测试没有匹配文件的情况"""
        # 设置模拟行为 - 目录存在但没有匹配文件
        mock_exists.return_value = True
        mock_glob.return_value = []
        
        # 创建CSV加载器并加载数据
        loader = CSVLoader(
            input_dir="/fake/path",
            file_pattern="nonexistent_*.csv"
        )
        result = loader.load()
        
        # 验证调用
        mock_exists.assert_called_with("/fake/path")
        mock_glob.assert_called_with(os.path.join("/fake/path", "nonexistent_*.csv"))
        mock_read_csv.assert_not_called()
        
        # 验证结果是空DataFrame
        self.assertTrue(result.empty)
    
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    def test_handle_read_error_with_mock(self, mock_read_csv, mock_glob, mock_exists):
        """使用mock测试读取错误处理"""
        # 设置模拟行为 - 第二个文件读取失败
        mock_exists.return_value = True
        mock_glob.return_value = [
            '/fake/path/data1.csv',
            '/fake/path/data2.csv'
        ]
        # 第一个文件读取成功，第二个文件引发异常
        df1 = self.test_data.copy()
        df1["source_file"] = "data1.csv"
        mock_read_csv.side_effect = [
            df1,
            pd.errors.EmptyDataError("No columns to parse from file")
        ]
        
        # 创建CSV加载器并加载数据
        loader = CSVLoader(
            input_dir="/fake/path",
            file_pattern="*.csv"
        )
        result = loader.load()
        
        # 验证调用
        self.assertEqual(mock_read_csv.call_count, 2)
        
        # 验证结果只包含第一个文件的数据
        pd.testing.assert_frame_equal(result, df1)
    
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('pandas.read_csv')
    def test_load_with_options_with_mock(self, mock_read_csv, mock_glob, mock_exists):
        """使用mock测试带选项的CSV加载"""
        # 设置模拟行为
        mock_exists.return_value = True
        mock_glob.return_value = ['/fake/path/data.csv']
        
        # 准备测试数据
        df = self.test_data.copy()
        df["source_file"] = "data.csv"
        mock_read_csv.return_value = df
        
        # 创建带有自定义选项的CSV加载器
        loader = CSVLoader(
            input_dir="/fake/path",
            file_pattern="*.csv",
            encoding="latin1",
            sep=";",
            index_col=0,
            usecols=["id", "value"]
        )
        result = loader.load()
        
        # 验证调用参数
        mock_read_csv.assert_called_once()
        args, kwargs = mock_read_csv.call_args
        self.assertEqual(args[0], '/fake/path/data.csv')
        self.assertEqual(kwargs.get('encoding'), 'latin1')
        self.assertEqual(kwargs.get('sep'), ';')
        self.assertEqual(kwargs.get('index_col'), 0)
        self.assertEqual(kwargs.get('usecols'), ["id", "value"])


# 使用pytest风格的测试
@pytest.mark.usefixtures("temp_dir", "temp_csv_files")
class TestCSVLoaderPytest:
    """使用pytest fixture测试CSVLoader类"""
    
    def test_load_csv_with_fixture(self, temp_dir, temp_csv_files, test_df):
        """使用pytest fixture测试加载CSV文件"""
        # 创建测试目录和文件
        input_dir = os.path.join(temp_dir, "csv_files")
        
        # 创建加载器并加载数据
        loader = CSVLoader(input_dir=input_dir, file_pattern="*.csv")
        result = loader.load()
        
        # 验证结果是否包含测试数据（两个文件合并）
        assert len(result) == len(test_df)  # 注意：temp_csv_files已经把test_df分成了两半
        assert set(test_df.columns).issubset(set(result.columns))
        assert "source_file" in result.columns
    
    def test_load_specific_pattern_with_fixture(self, temp_dir, temp_csv_files, test_df):
        """使用pytest fixture测试特定模式加载CSV文件"""
        # 创建测试目录和文件
        input_dir = os.path.join(temp_dir, "csv_files")
        
        # 创建一个特定的加载器，只加载data1.csv
        loader = CSVLoader(input_dir=input_dir, file_pattern="data1.csv")
        result = loader.load()
        
        # 验证结果只包含一个文件的数据
        assert len(result) == len(test_df) // 2
        assert set(test_df.columns).issubset(set(result.columns))
        assert "source_file" in result.columns
    
    def test_load_with_mock_fixture(self, mock_csv_loader):
        """使用mock fixture测试CSV加载器"""
        # 准备测试数据 
        df1 = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df2 = pd.DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
        
        # 添加source_file列模拟CSVLoader的行为
        df1_with_source = df1.copy()
        df1_with_source["source_file"] = "data1.csv"
        df2_with_source = df2.copy()
        df2_with_source["source_file"] = "data2.csv"
        
        # 设置mock文件和返回值的映射
        files = ["/fake/path/data1.csv", "/fake/path/data2.csv"]
        
        # 同时mock glob和pandas.read_csv
        with patch('glob.glob', return_value=files) as mock_glob, \
             patch('os.path.exists', return_value=True) as mock_exists, \
             patch('pandas.read_csv', side_effect=[df1_with_source, df2_with_source]) as mock_read_csv:
            
            # 创建加载器并加载数据
            loader = CSVLoader(input_dir="/fake/path", file_pattern="*.csv")
            result = loader.load()
            
            # 验证mock函数被正确调用
            mock_exists.assert_any_call("/fake/path")
            mock_glob.assert_called_once_with(os.path.join("/fake/path", "*.csv"))
            assert mock_read_csv.call_count == 2
            
            # 验证结果包含预期的数据
            assert len(result) == 4
            assert "col1" in result.columns
            assert "col2" in result.columns
            assert "source_file" in result.columns


class TestCustomLoader(unittest.TestCase):
    """测试自定义数据加载器"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.test_data = create_test_data(rows=5, columns=2)
        self.test_data.columns = ["id", "value"]

        # 定义自定义加载器
        class CustomLoader(DataLoader):
            def __init__(self, data=None, **kwargs):
                super().__init__(**kwargs)
                if data is None:
                    self.data = pd.DataFrame()
                else:
                    self.data = data

            def load(self):
                return self.data.copy()

        self.loader_class = CustomLoader
        
    def test_custom_loader(self):
        """测试自定义加载器"""
        # 创建自定义加载器实例
        loader = self.loader_class(
            data=self.test_data,
            name="custom_loader",
            description="自定义数据加载器"
        )
        
        # 加载数据
        result = loader.load()
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, self.test_data)


if __name__ == "__main__":
    unittest.main() 