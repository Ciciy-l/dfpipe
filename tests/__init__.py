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

def create_test_data(rows=10, columns=3, seed=42, add_id_column=False):
    """
    创建用于测试的DataFrame
    
    参数:
        rows (int): 数据行数
        columns (int): 数据列数
        seed (int): 随机种子
        add_id_column (bool): 是否添加ID列
    
    返回:
        pd.DataFrame: 测试数据框
    """
    np.random.seed(seed)
    
    # 创建测试数据
    data = np.random.randint(0, 100, size=(rows, columns))
    df = pd.DataFrame(data)
    
    # 设置列名
    col_names = [f"column_{i+1}" for i in range(columns)]
    df.columns = col_names
    
    # 可选添加ID列
    if add_id_column:
        df.insert(0, "id", range(1, rows + 1))
    
    return df 