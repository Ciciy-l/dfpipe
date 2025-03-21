#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from typing import List, Dict, Any

from dfpipe.core.base import DataProcessor
from dfpipe.core.registry import ComponentRegistry

"""
这个模块提供了几个基础处理器模板，方便用户扩展自定义处理器
"""

@ComponentRegistry.register_processor
class FilterProcessor(DataProcessor):
    """根据条件过滤数据"""
    
    def __init__(self, column: str, condition, **kwargs):
        """
        初始化过滤处理器
        
        Args:
            column: 要过滤的列名
            condition: 过滤条件，可以是一个函数或表达式
            **kwargs: 其他参数
        """
        super().__init__(
            name="FilterProcessor",
            description=f"根据列'{column}'过滤数据"
        )
        self.column = column
        self.condition = condition
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根据条件过滤数据
        
        Args:
            data: 输入数据框
            
        Returns:
            过滤后的数据框
        """
        if self.column not in data.columns:
            self.logger.warning(f"列'{self.column}'不存在于数据中，跳过过滤")
            return data
        
        original_count = len(data)
        
        try:
            # 应用过滤条件
            if callable(self.condition):
                result = data[data[self.column].apply(self.condition)]
            else:
                result = data[data[self.column] == self.condition]
            
            filtered_count = original_count - len(result)
            self.logger.info(f"过滤前: {original_count} 行, 过滤后: {len(result)} 行, 移除: {filtered_count} 行")
            
            return result
            
        except Exception as e:
            self.logger.error(f"过滤数据时出错: {str(e)}")
            return data


@ComponentRegistry.register_processor
class TransformProcessor(DataProcessor):
    """对指定列应用转换函数"""
    
    def __init__(self, column: str, transform_func, target_column: str = None, **kwargs):
        """
        初始化转换处理器
        
        Args:
            column: 要转换的列
            transform_func: 转换函数
            target_column: 结果存储列，如果不指定则覆盖原列
            **kwargs: 其他参数
        """
        target = target_column or column
        super().__init__(
            name="TransformProcessor",
            description=f"对列'{column}'应用转换，结果存储到'{target}'"
        )
        self.column = column
        self.transform_func = transform_func
        self.target_column = target_column
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用转换函数
        
        Args:
            data: 输入数据框
            
        Returns:
            转换后的数据框
        """
        if self.column not in data.columns:
            self.logger.warning(f"列'{self.column}'不存在于数据中，跳过转换")
            return data
        
        result = data.copy()
        
        try:
            # 应用转换
            target = self.target_column or self.column
            result[target] = result[self.column].apply(self.transform_func)
            
            self.logger.info(f"成功转换列'{self.column}'到'{target}'")
            
            return result
            
        except Exception as e:
            self.logger.error(f"转换列'{self.column}'时出错: {str(e)}")
            return data


@ComponentRegistry.register_processor
class ColumnProcessor(DataProcessor):
    """列操作处理器（添加、删除、重命名列）"""
    
    def __init__(self, operation: str, **kwargs):
        """
        初始化列操作处理器
        
        Args:
            operation: 操作类型，可选: 'add', 'drop', 'rename'
            **kwargs: 操作相关的参数
        """
        super().__init__(
            name="ColumnProcessor",
            description=f"对列进行{operation}操作"
        )
        self.operation = operation
        self.params = kwargs
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行列操作
        
        Args:
            data: 输入数据框
            
        Returns:
            处理后的数据框
        """
        result = data.copy()
        
        try:
            if self.operation == 'add':
                column = self.params.get('column')
                value = self.params.get('value')
                if column:
                    if callable(value):
                        result[column] = result.apply(value, axis=1)
                    else:
                        result[column] = value
                    self.logger.info(f"添加列: {column}")
                
            elif self.operation == 'drop':
                columns = self.params.get('columns', [])
                if isinstance(columns, str):
                    columns = [columns]
                
                existing_columns = [col for col in columns if col in result.columns]
                if existing_columns:
                    result = result.drop(columns=existing_columns)
                    self.logger.info(f"删除列: {', '.join(existing_columns)}")
                
            elif self.operation == 'rename':
                mapping = self.params.get('mapping', {})
                valid_mapping = {k: v for k, v in mapping.items() if k in result.columns}
                if valid_mapping:
                    result = result.rename(columns=valid_mapping)
                    renamed = [f"{k} -> {v}" for k, v in valid_mapping.items()]
                    self.logger.info(f"重命名列: {', '.join(renamed)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理列时出错: {str(e)}")
            return data 