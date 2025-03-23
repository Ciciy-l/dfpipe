#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dfpipe 测试包
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_test_data(rows=100, columns=5):
    """
    创建测试数据
    
    Args:
        rows: 行数
        columns: 列数
        
    Returns:
        pandas.DataFrame: 测试数据
    """
    np.random.seed(42)
    data = {
        f'col_{i}': np.random.randn(rows) for i in range(columns)
    }
    return pd.DataFrame(data) 