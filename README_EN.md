# DFPipe

A flexible and extensible DataFrame processing pipeline tool that supports multiple data sources, processing algorithms, and output formats.

## Features

- **Modular Design**: Provides three basic components: data loaders, processors, and writers
- **Flexible Configuration**: Supports building processing pipelines through JSON configuration files or code API
- **Easy to Extend**: Simple component registration mechanism for adding custom components
- **Rich Logging**: Detailed processing logs for debugging and monitoring

## Installation

```bash
# Install from PyPI
pip install dfpipe

# Install from source
git clone https://github.com/Ciciy-l/dfpipe.git
cd dfpipe
pip install -e .
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

## Component Overview

### Data Loaders

Data loaders are responsible for loading data from various sources. The default loader is `CSVLoader`.

#### Built-in Loaders

- **CSVLoader**: Loads data from CSV files
  - `input_dir`: Input directory
  - `file_pattern`: File matching pattern
  - `encoding`: File encoding

### Data Processors

Data processors are responsible for processing and transforming data.

#### Built-in Processors

- **FilterProcessor**: Filters data based on conditions
  - `column`: Column name
  - `condition`: Filter condition

- **TransformProcessor**: Applies transformation functions to columns
  - `column`: Column to transform
  - `transform_func`: Transformation function
  - `target_column`: Result storage column

- **ColumnProcessor**: Column operations (add, drop, rename)
  - `operation`: Operation type ('add', 'drop', 'rename')
  - Operation-specific parameters

### Data Writers

Data writers are responsible for saving processed data to various destinations.

#### Built-in Writers

- **CSVWriter**: Saves data as CSV files
  - `output_dir`: Output directory
  - `filename`: Filename
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