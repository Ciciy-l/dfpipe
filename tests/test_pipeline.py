#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import tempfile
import json
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open

from dfpipe.core.pipeline import Pipeline
from dfpipe.core.base import DataLoader, DataProcessor, DataWriter


class TestPipeline(unittest.TestCase):
    """测试Pipeline类的功能"""

    def setUp(self):
        """设置测试环境"""
        # 创建模拟对象
        self.mock_loader = MagicMock(spec=DataLoader)
        self.mock_loader.name = "MockLoader"
        self.mock_loader.load.return_value = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })

        self.mock_processor = MagicMock(spec=DataProcessor)
        self.mock_processor.name = "MockProcessor"
        self.mock_processor.process.return_value = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [100, 200, 300]
        })

        self.mock_writer = MagicMock(spec=DataWriter)
        self.mock_writer.name = "MockWriter"

        # 创建模拟注册表
        self.mock_registry = MagicMock()
        self.mock_registry.get_loader.return_value = self.mock_loader
        self.mock_registry.get_processor.return_value = self.mock_processor
        self.mock_registry.get_writer.return_value = self.mock_writer

    def test_pipeline_initialization(self):
        """测试管道初始化"""
        pipeline = Pipeline(name="TestPipeline")
        self.assertEqual(pipeline.name, "TestPipeline")
        self.assertIsNone(pipeline.loader)
        self.assertEqual(pipeline.processors, [])
        self.assertIsNone(pipeline.writer)

    def test_set_loader(self):
        """测试设置加载器"""
        pipeline = Pipeline()
        result = pipeline.set_loader(self.mock_loader)
        
        # 验证加载器已设置
        self.assertEqual(pipeline.loader, self.mock_loader)
        # 验证返回的是管道实例（链式调用）
        self.assertEqual(result, pipeline)

    def test_add_processor(self):
        """测试添加处理器"""
        pipeline = Pipeline()
        result = pipeline.add_processor(self.mock_processor)
        
        # 验证处理器已添加
        self.assertEqual(len(pipeline.processors), 1)
        self.assertEqual(pipeline.processors[0], self.mock_processor)
        # 验证返回的是管道实例（链式调用）
        self.assertEqual(result, pipeline)

    def test_add_multiple_processors(self):
        """测试添加多个处理器"""
        pipeline = Pipeline()
        
        # 创建多个处理器
        processor1 = MagicMock(spec=DataProcessor)
        processor1.name = "Processor1"
        processor2 = MagicMock(spec=DataProcessor)
        processor2.name = "Processor2"
        
        # 添加处理器
        pipeline.add_processor(processor1)
        pipeline.add_processor(processor2)
        
        # 验证处理器已添加且顺序正确
        self.assertEqual(len(pipeline.processors), 2)
        self.assertEqual(pipeline.processors[0], processor1)
        self.assertEqual(pipeline.processors[1], processor2)

    def test_set_writer(self):
        """测试设置写入器"""
        pipeline = Pipeline()
        result = pipeline.set_writer(self.mock_writer)
        
        # 验证写入器已设置
        self.assertEqual(pipeline.writer, self.mock_writer)
        # 验证返回的是管道实例（链式调用）
        self.assertEqual(result, pipeline)

    def test_validate_success(self):
        """测试验证成功"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        pipeline.set_writer(self.mock_writer)
        
        # 验证通过
        self.assertTrue(pipeline.validate())

    def test_validate_no_loader(self):
        """测试验证失败 - 无加载器"""
        pipeline = Pipeline()
        pipeline.set_writer(self.mock_writer)
        
        # 验证失败 - 无加载器
        self.assertFalse(pipeline.validate())

    def test_validate_no_writer(self):
        """测试验证失败 - 无写入器"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        
        # 验证失败 - 无写入器
        self.assertFalse(pipeline.validate())

    def test_run_success(self):
        """测试正常运行管道"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        pipeline.add_processor(self.mock_processor)
        pipeline.set_writer(self.mock_writer)
        
        # 运行管道
        result = pipeline.run()
        
        # 验证结果
        self.assertTrue(result)
        self.mock_loader.load.assert_called_once()
        self.mock_processor.process.assert_called_once()
        self.mock_writer.write.assert_called_once()

    def test_run_validation_fails(self):
        """测试运行 - 验证失败"""
        pipeline = Pipeline()
        # 故意不设置加载器和写入器
        
        # 运行管道应该在验证时失败
        result = pipeline.run()
        
        # 验证结果
        self.assertFalse(result)
        # 验证加载器和处理器没有被调用
        self.mock_loader.load.assert_not_called()
        self.mock_processor.process.assert_not_called()
        self.mock_writer.write.assert_not_called()

    def test_run_empty_data(self):
        """测试运行 - 空数据"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        pipeline.add_processor(self.mock_processor)
        pipeline.set_writer(self.mock_writer)
        
        # 模拟加载器返回空数据
        empty_df = pd.DataFrame()
        self.mock_loader.load.return_value = empty_df
        
        # 运行管道
        result = pipeline.run()
        
        # 验证结果 - 应该成功但没有处理和写入
        self.assertTrue(result)
        self.mock_loader.load.assert_called_once()
        self.mock_processor.process.assert_not_called()
        self.mock_writer.write.assert_not_called()

    def test_run_processor_error(self):
        """测试运行 - 处理器错误"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        pipeline.add_processor(self.mock_processor)
        pipeline.set_writer(self.mock_writer)
        
        # 模拟处理器引发异常
        self.mock_processor.process.side_effect = Exception("处理器错误")
        
        # 运行管道
        result = pipeline.run()
        
        # 验证结果 - 应该失败
        self.assertFalse(result)
        self.mock_loader.load.assert_called_once()
        self.mock_processor.process.assert_called_once()
        self.mock_writer.write.assert_not_called()

    def test_run_loader_error(self):
        """测试运行 - 加载器错误"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        pipeline.add_processor(self.mock_processor)
        pipeline.set_writer(self.mock_writer)
        
        # 模拟加载器引发异常
        self.mock_loader.load.side_effect = Exception("加载器错误")
        
        # 运行管道
        result = pipeline.run()
        
        # 验证结果 - 应该失败
        self.assertFalse(result)
        self.mock_loader.load.assert_called_once()
        self.mock_processor.process.assert_not_called()
        self.mock_writer.write.assert_not_called()

    def test_run_writer_error(self):
        """测试运行 - 写入器错误"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        pipeline.add_processor(self.mock_processor)
        pipeline.set_writer(self.mock_writer)
        
        # 模拟写入器引发异常
        self.mock_writer.write.side_effect = Exception("写入器错误")
        
        # 运行管道
        result = pipeline.run()
        
        # 验证结果 - 应该失败
        self.assertFalse(result)
        self.mock_loader.load.assert_called_once()
        self.mock_processor.process.assert_called_once()
        self.mock_writer.write.assert_called_once()

    def test_processor_empty_result(self):
        """测试处理器返回空数据"""
        pipeline = Pipeline()
        pipeline.set_loader(self.mock_loader)
        pipeline.add_processor(self.mock_processor)
        pipeline.set_writer(self.mock_writer)
        
        # 模拟处理器返回空数据
        self.mock_processor.process.return_value = pd.DataFrame()
        
        # 运行管道
        result = pipeline.run()
        
        # 验证结果 - 应该成功但没有写入
        self.assertTrue(result)
        self.mock_loader.load.assert_called_once()
        self.mock_processor.process.assert_called_once()
        self.mock_writer.write.assert_not_called()

    def test_from_config(self):
        """测试从配置创建管道"""
        config = {
            "name": "ConfigPipeline",
            "loader": {
                "name": "CSVLoader",
                "params": {"input_dir": "data"}
            },
            "processors": [
                {
                    "name": "FilterProcessor",
                    "params": {"column": "age", "condition": 18}
                }
            ],
            "writer": {
                "name": "CSVWriter",
                "params": {"output_dir": "output"}
            }
        }
        
        # 创建管道
        pipeline = Pipeline.from_config(config, self.mock_registry)
        
        # 验证结果
        self.assertEqual(pipeline.name, "ConfigPipeline")
        self.mock_registry.get_loader.assert_called_once_with("CSVLoader", input_dir="data")
        self.mock_registry.get_processor.assert_called_once_with("FilterProcessor", column="age", condition=18)
        self.mock_registry.get_writer.assert_called_once_with("CSVWriter", output_dir="output")

    def test_from_config_loader_error(self):
        """测试配置加载器错误"""
        config = {
            "name": "ConfigPipeline",
            "loader": {
                "name": "InvalidLoader",
                "params": {}
            }
        }
        
        # 模拟注册表引发异常
        self.mock_registry.get_loader.side_effect = Exception("无效加载器")
        
        # 创建管道应该引发异常
        with self.assertRaises(Exception):
            Pipeline.from_config(config, self.mock_registry)

    def test_from_config_processor_error(self):
        """测试配置处理器错误"""
        config = {
            "name": "ConfigPipeline",
            "loader": {
                "name": "CSVLoader",
                "params": {}
            },
            "processors": [
                {
                    "name": "InvalidProcessor",
                    "params": {}
                }
            ]
        }
        
        # 模拟注册表引发异常
        self.mock_registry.get_processor.side_effect = Exception("无效处理器")
        
        # 创建管道应该引发异常
        with self.assertRaises(Exception):
            Pipeline.from_config(config, self.mock_registry)

    def test_from_config_writer_error(self):
        """测试配置写入器错误"""
        config = {
            "name": "ConfigPipeline",
            "loader": {
                "name": "CSVLoader",
                "params": {}
            },
            "writer": {
                "name": "InvalidWriter",
                "params": {}
            }
        }
        
        # 模拟注册表引发异常
        self.mock_registry.get_writer.side_effect = Exception("无效写入器")
        
        # 创建管道应该引发异常
        with self.assertRaises(Exception):
            Pipeline.from_config(config, self.mock_registry)

    def test_from_config_no_loader(self):
        """测试配置中没有加载器的情况"""
        config = {
            "name": "NoLoaderPipeline",
            "writer": {
                "name": "CSVWriter",
                "params": {}
            }
        }
        
        # 创建没有loader的管道
        pipeline = Pipeline.from_config(config, self.mock_registry)
        
        # 验证结果
        self.assertEqual(pipeline.name, "NoLoaderPipeline")
        self.assertIsNone(pipeline.loader)
        self.mock_registry.get_loader.assert_not_called()
        self.mock_registry.get_writer.assert_called_once_with("CSVWriter")

    def test_from_config_no_writer(self):
        """测试配置中没有写入器的情况"""
        config = {
            "name": "NoWriterPipeline",
            "loader": {
                "name": "CSVLoader",
                "params": {}
            }
        }
        
        # 创建没有writer的管道
        pipeline = Pipeline.from_config(config, self.mock_registry)
        
        # 验证结果
        self.assertEqual(pipeline.name, "NoWriterPipeline")
        self.assertIsNone(pipeline.writer)
        self.mock_registry.get_loader.assert_called_once_with("CSVLoader")
        self.mock_registry.get_writer.assert_not_called()

    def test_from_config_empty_processor_name(self):
        """测试处理器名称为空的情况"""
        config = {
            "name": "EmptyProcessorNamePipeline",
            "processors": [
                {
                    "name": "",  # 空名称
                    "params": {}
                }
            ]
        }
        
        # 创建管道
        pipeline = Pipeline.from_config(config, self.mock_registry)
        
        # 验证结果
        self.assertEqual(pipeline.name, "EmptyProcessorNamePipeline")
        self.assertEqual(len(pipeline.processors), 0)  # 不应该添加处理器
        self.mock_registry.get_processor.assert_not_called()

    @patch('builtins.open', new_callable=mock_open, read_data='{"name": "JsonPipeline"}')
    def test_from_json(self, mock_file):
        """测试从JSON创建管道"""
        # 模拟配置数据
        config_data = {
            "name": "JsonPipeline",
            "loader": {"name": "CSVLoader", "params": {}},
            "writer": {"name": "CSVWriter", "params": {}}
        }
        
        # 模拟json.load返回配置数据
        with patch('json.load', return_value=config_data):
            # 创建管道
            pipeline = Pipeline.from_json("config.json", self.mock_registry)
            
            # 验证结果
            self.assertEqual(pipeline.name, "JsonPipeline")
            mock_file.assert_called_once_with("config.json", 'r', encoding='utf-8')

    @patch('builtins.open', side_effect=IOError("文件不存在"))
    def test_from_json_file_error(self, mock_file):
        """测试从JSON创建 - 文件错误"""
        # 创建管道应该引发异常
        with self.assertRaises(Exception):
            Pipeline.from_json("invalid.json", self.mock_registry)

    def test_chain_calls(self):
        """测试链式调用"""
        pipeline = Pipeline()
        
        # 链式设置组件
        result = pipeline.set_loader(self.mock_loader) \
                        .add_processor(self.mock_processor) \
                        .set_writer(self.mock_writer)
        
        # 验证结果
        self.assertEqual(result, pipeline)
        self.assertEqual(pipeline.loader, self.mock_loader)
        self.assertEqual(pipeline.processors[0], self.mock_processor)
        self.assertEqual(pipeline.writer, self.mock_writer)

    def test_multiple_processors_run_order(self):
        """测试多个处理器按顺序运行"""
        pipeline = Pipeline()
        
        # 创建两个处理器
        processor1 = MagicMock(spec=DataProcessor)
        processor1.name = "Processor1"
        processor1.process.return_value = pd.DataFrame({
            "id": [1, 2],
            "step": ["A", "A"]
        })
        
        processor2 = MagicMock(spec=DataProcessor)
        processor2.name = "Processor2"
        processor2.process.return_value = pd.DataFrame({
            "id": [1, 2],
            "step": ["B", "B"]
        })
        
        # 设置管道
        pipeline.set_loader(self.mock_loader)
        pipeline.add_processor(processor1)
        pipeline.add_processor(processor2)
        pipeline.set_writer(self.mock_writer)
        
        # 运行管道
        pipeline.run()
        
        # 验证处理器按顺序调用
        processor1.process.assert_called_once()
        processor2.process.assert_called_once()
        
        # 验证第二个处理器接收第一个处理器的输出
        pd.testing.assert_frame_equal(
            processor2.process.call_args[0][0],
            processor1.process.return_value
        )


if __name__ == "__main__":
    unittest.main() 