#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pytest配置文件
定义所有共享的测试fixtures
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime

from tests import create_test_data


@pytest.fixture
def test_df():
    """返回测试用的DataFrame"""
    return pd.DataFrame({
        "id": range(1, 11),
        "value": np.random.randint(10, 100, 10),
        "category": [f"category_{i}" for i in "ABCDEABCDE"]
    })


@pytest.fixture
def temp_dir(tmpdir):
    """创建临时目录，用于测试文件操作"""
    return str(tmpdir)


@pytest.fixture
def temp_csv_files(temp_dir, test_df):
    """创建临时CSV文件用于测试"""
    # 创建输入目录
    csv_dir = os.path.join(temp_dir, "csv_files")
    os.makedirs(csv_dir, exist_ok=True)
    
    # 将测试数据分成两半，存入两个CSV文件
    half_size = len(test_df) // 2
    test_df.iloc[:half_size].to_csv(os.path.join(csv_dir, "data1.csv"), index=False)
    test_df.iloc[half_size:].to_csv(os.path.join(csv_dir, "data2.csv"), index=False)
    
    return csv_dir


@pytest.fixture
def mock_csv_loader():
    """模拟CSV加载器的文件系统操作
    
    Args:
        exists: 目录是否存在
        files: glob.glob应返回的文件列表
        dataframes: pandas.read_csv应返回的DataFrame列表
    """
    def _create_mock(exists=True, files=None, dataframes=None):
        # 创建模拟对象
        mock_exists = MagicMock(return_value=exists)
        mock_glob = MagicMock(return_value=files or [])
        
        # 为read_csv创建特殊的mock，根据文件名返回不同的DataFrame
        mock_read_csv = MagicMock()
        
        if files and dataframes:
            # 确保文件和数据数量匹配
            assert len(files) == len(dataframes), "文件数量和DataFrame数量必须一致"
            
            # 创建side_effect函数根据文件名返回对应的DataFrame
            file_to_df = {file: df for file, df in zip(files, dataframes)}
            
            def read_csv_side_effect(file_path, **kwargs):
                # 为每个读取的文件添加source_file列
                df = file_to_df.get(file_path).copy()
                df['source_file'] = os.path.basename(file_path)
                return df
                
            mock_read_csv.side_effect = read_csv_side_effect
        
        return mock_glob, mock_read_csv, mock_exists
    
    return _create_mock


@pytest.fixture
def mock_csv_writer():
    """模拟CSV写入器的文件系统操作
    
    Args:
        exists: 目录是否存在
    """
    def _create_mock(exists=True):
        # 创建模拟对象
        mock_exists = MagicMock(return_value=exists)
        mock_makedirs = MagicMock()
        mock_to_csv = MagicMock()
        
        return mock_to_csv, mock_makedirs, mock_exists
    
    return _create_mock


@pytest.fixture
def mock_datetime():
    """模拟日期时间"""
    with patch('datetime.datetime') as mock:
        # 设置now()返回的模拟对象
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20230101_120000"
        mock.now.return_value = mock_now
        yield mock 