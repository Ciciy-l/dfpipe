#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import tempfile
import importlib
import sys
from unittest.mock import patch, MagicMock, call
import inspect

import pandas as pd

from dfpipe.core.registry import ComponentRegistry
from dfpipe.core.base import DataLoader, DataProcessor, DataWriter


class TestComponentRegistry(unittest.TestCase):
    """测试组件注册模块"""

    def setUp(self):
        """设置测试环境"""
        # 保存原始注册表状态
        self.original_loaders = ComponentRegistry._loaders.copy()
        self.original_processors = ComponentRegistry._processors.copy()
        self.original_writers = ComponentRegistry._writers.copy()
        
        # 创建测试组件类
        class TestLoader(DataLoader):
            def __init__(self, test_param=None):
                self.test_param = test_param
            
            def load(self):
                return pd.DataFrame({'test': [1, 2, 3]})
        
        class TestProcessor(DataProcessor):
            def __init__(self, test_param=None):
                self.test_param = test_param
            
            def process(self, data):
                return data
        
        class TestWriter(DataWriter):
            def __init__(self, test_param=None):
                self.test_param = test_param
            
            def write(self, data):
                pass
        
        # 保存测试类引用
        self.TestLoader = TestLoader
        self.TestProcessor = TestProcessor
        self.TestWriter = TestWriter

    def tearDown(self):
        """清理测试环境"""
        # 恢复原始注册表状态
        ComponentRegistry._loaders = self.original_loaders.copy()
        ComponentRegistry._processors = self.original_processors.copy()
        ComponentRegistry._writers = self.original_writers.copy()

    def test_register_loader(self):
        """测试注册加载器"""
        # 注册测试加载器
        result = ComponentRegistry.register_loader(self.TestLoader)
        
        # 验证返回的类是否正确
        self.assertEqual(result, self.TestLoader)
        
        # 验证加载器是否已注册
        self.assertIn('TestLoader', ComponentRegistry._loaders)
        self.assertEqual(ComponentRegistry._loaders['TestLoader'], self.TestLoader)

    def test_register_processor(self):
        """测试注册处理器"""
        # 注册测试处理器
        result = ComponentRegistry.register_processor(self.TestProcessor)
        
        # 验证返回的类是否正确
        self.assertEqual(result, self.TestProcessor)
        
        # 验证处理器是否已注册
        self.assertIn('TestProcessor', ComponentRegistry._processors)
        self.assertEqual(ComponentRegistry._processors['TestProcessor'], self.TestProcessor)

    def test_register_writer(self):
        """测试注册写入器"""
        # 注册测试写入器
        result = ComponentRegistry.register_writer(self.TestWriter)
        
        # 验证返回的类是否正确
        self.assertEqual(result, self.TestWriter)
        
        # 验证写入器是否已注册
        self.assertIn('TestWriter', ComponentRegistry._writers)
        self.assertEqual(ComponentRegistry._writers['TestWriter'], self.TestWriter)

    def test_get_loader(self):
        """测试获取加载器实例"""
        # 注册测试加载器
        ComponentRegistry.register_loader(self.TestLoader)
        
        # 获取加载器实例
        loader = ComponentRegistry.get_loader('TestLoader', test_param='value')
        
        # 验证实例类型和参数
        self.assertIsInstance(loader, self.TestLoader)
        self.assertEqual(loader.test_param, 'value')
        
        # 测试获取不存在的加载器
        with self.assertRaises(ValueError):
            ComponentRegistry.get_loader('NonExistentLoader')

    def test_get_processor(self):
        """测试获取处理器实例"""
        # 注册测试处理器
        ComponentRegistry.register_processor(self.TestProcessor)
        
        # 获取处理器实例
        processor = ComponentRegistry.get_processor('TestProcessor', test_param='value')
        
        # 验证实例类型和参数
        self.assertIsInstance(processor, self.TestProcessor)
        self.assertEqual(processor.test_param, 'value')
        
        # 测试获取不存在的处理器
        with self.assertRaises(ValueError):
            ComponentRegistry.get_processor('NonExistentProcessor')

    def test_get_writer(self):
        """测试获取写入器实例"""
        # 注册测试写入器
        ComponentRegistry.register_writer(self.TestWriter)
        
        # 获取写入器实例
        writer = ComponentRegistry.get_writer('TestWriter', test_param='value')
        
        # 验证实例类型和参数
        self.assertIsInstance(writer, self.TestWriter)
        self.assertEqual(writer.test_param, 'value')
        
        # 测试获取不存在的写入器
        with self.assertRaises(ValueError):
            ComponentRegistry.get_writer('NonExistentWriter')

    def test_list_loaders(self):
        """测试列出所有加载器"""
        # 清空当前加载器
        ComponentRegistry._loaders.clear()
        
        # 注册测试加载器
        ComponentRegistry.register_loader(self.TestLoader)
        
        # 验证列出的加载器
        loaders = ComponentRegistry.list_loaders()
        self.assertEqual(loaders, ['TestLoader'])

    def test_list_processors(self):
        """测试列出所有处理器"""
        # 清空当前处理器
        ComponentRegistry._processors.clear()
        
        # 注册测试处理器
        ComponentRegistry.register_processor(self.TestProcessor)
        
        # 验证列出的处理器
        processors = ComponentRegistry.list_processors()
        self.assertEqual(processors, ['TestProcessor'])

    def test_list_writers(self):
        """测试列出所有写入器"""
        # 清空当前写入器
        ComponentRegistry._writers.clear()
        
        # 注册测试写入器
        ComponentRegistry.register_writer(self.TestWriter)
        
        # 验证列出的写入器
        writers = ComponentRegistry.list_writers()
        self.assertEqual(writers, ['TestWriter'])

    def test_auto_discover(self):
        """测试自动发现组件"""
        # 清空注册表，确保测试结果可预测
        ComponentRegistry._loaders.clear()
        ComponentRegistry._processors.clear()
        ComponentRegistry._writers.clear()
        
        # 为了避免影响其他测试以及真实导入产生的副作用，使用更精确的模拟路径
        with patch('dfpipe.core.registry.os.path.dirname', return_value='/fake/path'), \
             patch('dfpipe.core.registry.os.path.basename', side_effect=lambda x: x.split('/')[-1]), \
             patch('dfpipe.core.registry.os.path.exists', return_value=True), \
             patch('dfpipe.core.registry.os.path.join', side_effect=lambda *args: '/'.join(args)), \
             patch('dfpipe.core.registry.os.listdir', return_value=['test_loader.py']), \
             patch('dfpipe.core.registry.importlib.import_module') as mock_import:
            
            # 创建真实的测试类
            class TestLoader(DataLoader):
                def load(self):
                    return pd.DataFrame()
            
            # 配置mock_import以返回真实的模块
            def mock_import_func(name):
                mock_module = MagicMock()
                mock_module.TestLoader = TestLoader
                # 为 inspect.getmembers 提供必要的属性
                mock_module.__all__ = ['TestLoader']
                mock_module.__dict__ = {'TestLoader': TestLoader}
                return mock_module
                
            mock_import.side_effect = mock_import_func
            
            # 直接模拟ComponentRegistry._discover_components方法，避免复杂的嵌套模拟
            original_discover = ComponentRegistry._discover_components
            
            def mock_discover(cls, directory, base_class, register_func):
                if 'loaders' in directory:
                    register_func(TestLoader)
            
            # 替换方法
            ComponentRegistry._discover_components = classmethod(mock_discover)
            
            try:
                # 调用auto_discover
                ComponentRegistry.auto_discover()
                
                # 验证结果
                self.assertIn('TestLoader', ComponentRegistry._loaders)
            finally:
                # 恢复原方法
                ComponentRegistry._discover_components = original_discover

    def test_discover_components_directory_not_exists(self):
        """测试目录不存在时的组件发现"""
        with patch('os.path.exists', return_value=False), \
             patch('dfpipe.core.registry.logger.warning') as mock_warning:
            
            # 调用_discover_components
            ComponentRegistry._discover_components('/fake/path', DataLoader, ComponentRegistry.register_loader)
            
            # 验证警告日志
            mock_warning.assert_called_once()

    def test_discover_components_import_error(self):
        """测试导入错误时的组件发现"""
        # 直接验证日志输出而不是模拟对象调用
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['error_module.py']), \
             patch('dfpipe.core.registry.importlib.import_module', side_effect=ImportError("测试导入错误")), \
             self.assertLogs(logger='DataPipeline.Registry', level='ERROR') as log_ctx:
            
            # 调用_discover_components
            ComponentRegistry._discover_components('/fake/path', DataLoader, ComponentRegistry.register_loader)
            
            # 验证日志内容
            self.assertEqual(len(log_ctx.records), 1)
            self.assertIn('导入模块', log_ctx.output[0])
            self.assertIn('测试导入错误', log_ctx.output[0])

    def test_decorator_registration(self):
        """测试装饰器注册方式"""
        # 清空当前注册表
        ComponentRegistry._loaders.clear()
        ComponentRegistry._processors.clear()
        ComponentRegistry._writers.clear()
        
        # 使用装饰器注册组件
        @ComponentRegistry.register_loader
        class DecoratedLoader(DataLoader):
            def load(self):
                return pd.DataFrame()
        
        @ComponentRegistry.register_processor
        class DecoratedProcessor(DataProcessor):
            def process(self, data):
                return data
        
        @ComponentRegistry.register_writer
        class DecoratedWriter(DataWriter):
            def write(self, data):
                pass
        
        # 验证注册结果
        self.assertIn('DecoratedLoader', ComponentRegistry._loaders)
        self.assertIn('DecoratedProcessor', ComponentRegistry._processors)
        self.assertIn('DecoratedWriter', ComponentRegistry._writers)


if __name__ == "__main__":
    unittest.main() 