#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础组件测试模块
包含dfpipe核心基础组件(加载器、处理器基类、输出器)的单元测试
"""

import unittest

import pandas as pd

from dfpipe.core.base import DataLoader, DataProcessor, DataWriter
from tests import create_test_data


class TestDataLoader(unittest.TestCase):
    """测试数据加载器基类"""

    def setUp(self):
        class MockLoader(DataLoader):
            def load(self):
                return create_test_data()

        self.loader = MockLoader(name="test_loader", description="测试加载器")

    def test_loader_initialization(self):
        """测试加载器初始化"""
        self.assertEqual(self.loader.name, "test_loader")
        self.assertEqual(self.loader.description, "测试加载器")

    def test_loader_load(self):
        """测试加载器加载数据"""
        data = self.loader.load()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_loader_str(self):
        """测试加载器字符串表示"""
        expected_str = "test_loader: 测试加载器"
        self.assertEqual(str(self.loader), expected_str)

    def test_abstract_methods(self):
        """测试抽象方法是否必须实现"""
        with self.assertRaises(TypeError):
            # 创建一个不实现抽象方法的子类，应该会引发TypeError
            class InvalidLoader(DataLoader):
                pass

            # 尝试实例化应该会失败
            invalid_loader = InvalidLoader(name="invalid", description="无效加载器")


class TestDataProcessor(unittest.TestCase):
    """测试数据处理器基类"""

    def setUp(self):
        class MockProcessor(DataProcessor):
            def process(self, data):
                return data * 2

        self.processor = MockProcessor(name="test_processor", description="测试处理器")
        self.test_data = create_test_data()

    def test_processor_initialization(self):
        """测试处理器初始化"""
        self.assertEqual(self.processor.name, "test_processor")
        self.assertEqual(self.processor.description, "测试处理器")

    def test_processor_process(self):
        """测试处理器处理数据"""
        result = self.processor.process(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.test_data.shape)
        pd.testing.assert_frame_equal(result, self.test_data * 2)

    def test_processor_str(self):
        """测试处理器字符串表示"""
        expected_str = "test_processor: 测试处理器"
        self.assertEqual(str(self.processor), expected_str)

    def test_abstract_methods(self):
        """测试抽象方法是否必须实现"""
        with self.assertRaises(TypeError):
            # 创建一个不实现抽象方法的子类，应该会引发TypeError
            class InvalidProcessor(DataProcessor):
                pass

            # 尝试实例化应该会失败
            invalid_processor = InvalidProcessor(name="invalid", description="无效处理器")


class TestDataWriter(unittest.TestCase):
    """测试数据输出器基类"""

    def setUp(self):
        class MockWriter(DataWriter):
            def write(self, data):
                self.written_data = data

        self.writer = MockWriter(name="test_writer", description="测试输出器")
        self.test_data = create_test_data()

    def test_writer_initialization(self):
        """测试输出器初始化"""
        self.assertEqual(self.writer.name, "test_writer")
        self.assertEqual(self.writer.description, "测试输出器")

    def test_writer_write(self):
        """测试输出器写入数据"""
        self.writer.write(self.test_data)
        self.assertIsInstance(self.writer.written_data, pd.DataFrame)
        pd.testing.assert_frame_equal(self.writer.written_data, self.test_data)

    def test_writer_str(self):
        """测试输出器字符串表示"""
        expected_str = "test_writer: 测试输出器"
        self.assertEqual(str(self.writer), expected_str)

    def test_abstract_methods(self):
        """测试抽象方法是否必须实现"""
        with self.assertRaises(TypeError):
            # 创建一个不实现抽象方法的子类，应该会引发TypeError
            class InvalidWriter(DataWriter):
                pass

            # 尝试实例化应该会失败
            invalid_writer = InvalidWriter(name="invalid", description="无效输出器")


if __name__ == "__main__":
    unittest.main()
