# DFPipe

[![English](https://img.shields.io/badge/English-007ACC?style=flat-square&logo=python&logoColor=white)](README_EN.md) [![中文](https://img.shields.io/badge/中文-4A4A4A?style=flat-square&logo=python&logoColor=white)](README.md)

A flexible and extensible DataFrame processing pipeline tool that supports multiple data sources, processing algorithms, and output formats.

## Features

- **Modular Design**: Provides three basic components: data loader, processor, and writer
- **Flexible Configuration**: Supports building processing workflows through JSON configuration files or code API
- **Easy to Extend**: Simple component registration mechanism for adding custom components
- **Rich Logging**: Detailed processing logs for debugging and monitoring
- **Wide Compatibility**: Supports all Python versions from 3.6 to 3.12

## System Requirements

- Python >= 3.6
- pandas >= 1.3.0
- numpy >= 1.20.0

## Test Coverage

DFPipe maintains a high level of test coverage to ensure reliability and stability:

- **Comprehensive Test Suite**: Unit tests for all core components and integration tests for end-to-end workflows
- **Mock-based Testing**: Efficient testing using mock objects to simulate file systems and external dependencies
- **CI/CD Integration**: Automated testing on multiple Python versions to ensure cross-version compatibility

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=dfpipe

# Generate HTML coverage report
pytest --cov=dfpipe --cov-report=html
```

The current test coverage is over 99% for core components, ensuring robust behavior in various scenarios.

## Installation

```bash
# Install from PyPI
pip install dfpipe

# Install from source
git clone https://github.com/Ciciy-l/dfpipe.git
cd dfpipe
pip install -e .

# Install development version (includes testing tools)
pip install -e ".[dev]"
```

## Quick Start

### Using Command Line

The simplest way to use is through the command line:

```bash
python -m dfpipe --input-dir data --output-dir output
```

This will process data using the default CSV loader and writer.

### Using Configuration File

Create a configuration file to define a complete data processing pipeline:

```bash
python -m dfpipe --config dfpipe/examples/simple.json
```

### Configuration File Example

```json
{
    "name": "Simple Data Processing Pipeline",
    "loader": {
        "name": "CSVLoader",
        "params": {
            "input_dir": "data",
            "file_pattern": "*.csv"
        }
    },
    "processors": [
        {
            "name": "FilterProcessor",
            "params": {
                "column": "age",
                "condition": 18
            }
        }
    ],
    "writer": {
        "name": "CSVWriter",
        "params": {
            "output_dir": "output",
            "filename": "processed_data.csv"
        }
    }
}
```

## Using in Code

You can build and execute pipelines using the API in Python code:

```python
from dfpipe import Pipeline, ComponentRegistry, setup_logging

# Setup logging
logger = setup_logging()

# Initialize component registry
ComponentRegistry.auto_discover()

# Create pipeline
pipeline = Pipeline(name="MyPipeline")

# Set loader
loader = ComponentRegistry.get_loader("CSVLoader", input_dir="data")
pipeline.set_loader(loader)

# Add processor
filter_processor = ComponentRegistry.get_processor("FilterProcessor", column="age", condition=18)
pipeline.add_processor(filter_processor)

# Set writer
writer = ComponentRegistry.get_writer("CSVWriter", output_dir="output")
pipeline.set_writer(writer)

# Run pipeline
pipeline.run()
```

## Component Introduction

### Data Loader

Data loaders are responsible for loading data from various data sources. The default loader is `CSVLoader`.

#### Built-in Loaders

- **CSVLoader**: Load data from CSV files
  - `input_dir`: Input directory
  - `file_pattern`: File matching pattern
  - `encoding`: File encoding

### Data Processor

Data processors are responsible for processing and transforming data.

#### Built-in Processors

- **FilterProcessor**: Filter data based on conditions
  - `column`: Column name
  - `condition`: Filter condition

- **TransformProcessor**: Apply transformation function to columns
  - `column`: Column to transform
  - `transform_func`: Transformation function
  - `target_column`: Result storage column

- **ColumnProcessor**: Column operations (add, drop, rename)
  - `operation`: Operation type ('add', 'drop', 'rename')
  - Operation-specific parameters

- **FieldsOrganizer**: Organize and standardize fields
  - `target_columns`: List of target fields, result will only contain these fields in the specified order
  - `default_values`: Dictionary of field default values, key is field name, value is default value. If not provided, empty string is used by default

### Data Writer

Data writers are responsible for saving processed data to various destinations.

#### Built-in Writers

- **CSVWriter**: Save data as CSV files
  - `output_dir`: Output directory
  - `filename`: File name
  - `encoding`: File encoding

## Custom Components

### Creating Custom Loader

```python
from dfpipe import DataLoader, ComponentRegistry

@ComponentRegistry.register_loader
class MyCustomLoader(DataLoader):
    def __init__(self, param1, param2=None, **kwargs):
        super().__init__(
            name="MyCustomLoader",
            description="My custom loader"
        )
        self.param1 = param1
        self.param2 = param2

    def load(self) -> pd.DataFrame:
        # Implement loading logic
        # ...
        return data_frame
```

### Creating Custom Processor

```python
from dfpipe import DataProcessor, ComponentRegistry

@ComponentRegistry.register_processor
class MyCustomProcessor(DataProcessor):
    def __init__(self, param1, param2=None, **kwargs):
        super().__init__(
            name="MyCustomProcessor",
            description="My custom processor"
        )
        self.param1 = param1
        self.param2 = param2

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implement processing logic
        # ...
        return processed_data
```

### Creating Custom Writer

```python
from dfpipe import DataWriter, ComponentRegistry

@ComponentRegistry.register_writer
class MyCustomWriter(DataWriter):
    def __init__(self, param1, param2=None, **kwargs):
        super().__init__(
            name="MyCustomWriter",
            description="My custom writer"
        )
        self.param1 = param1
        self.param2 = param2

    def write(self, data: pd.DataFrame) -> None:
        # Implement writing logic
        # ...
```

## License

MIT

## Developer Guide

### Code Style Guidelines

This project uses the following tools to maintain code style consistency:

- **Black**: Automatic code formatting tool
- **isort**: Import statement sorting tool

#### Local Setup

1. Install development dependencies:

```bash
pip install black isort pre-commit
```

2. Set up pre-commit hooks:

```bash
pre-commit install
```

This will automatically run code formatting checks before each commit.

#### Manual Formatting

```bash
# Format all Python files
black .

# Sort imports in all Python files
isort .
```

#### Configuration Files

Code style configurations are defined in the `pyproject.toml` file.

### Continuous Integration

We use GitHub Actions for continuous integration, ensuring all commits pass tests and code style checks. The workflow configuration can be found in `.github/workflows/python-test.yml`.
