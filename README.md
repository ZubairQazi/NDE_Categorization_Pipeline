# Topic Categorization Pipeline

## Overview

This project is a data processing pipeline designed to categorize scientific datasets using large language models. It includes components for data input, processing, and output, allowing users to efficiently manage and analyze their data.

The pipeline utilizes base classes for input, processing, categorization, and output, maintaining a high level of flexibility and extensibility that allows users to customize and adapt components to their specific needs.

## Processing Modes

The pipeline offers two distinct processing modes to accommodate different use cases and reliability requirements:

### Sync Mode with Checkpointing

Sync mode processes records individually in real-time with built-in checkpointing and crash recovery capabilities. This mode is ideal for:
- Long-running processing jobs that need reliability
- Scenarios where you want to see progress and results in real-time
- Cases where you need to resume processing after interruptions

#### Key Features:
- **Record-by-record processing**: Each item is processed individually, not in batches
- **Automatic checkpointing**: Save progress every N records (configurable)
- **Crash recovery**: Resume from the last checkpoint if processing is interrupted
- **Real-time output**: Results are written immediately as they're processed
- **Progress tracking**: Live progress updates with ETA estimates

#### Usage Examples:

**Basic sync mode**:
```bash
python nde_pipeline_topic_categories.py --mode sync --dataset_path data.json --dataset_name zenodo
```

**With custom checkpoint interval** (checkpoint every 25 records):
```bash
python nde_pipeline_topic_categories.py --mode sync --dataset_path data.json --dataset_name zenodo --checkpoint_interval 25
```

**Resume from checkpoint**:
```bash
python nde_pipeline_topic_categories.py --mode sync --resume --session_id sync_20240728_143022
```

**List available checkpoint sessions**:
```bash
python nde_pipeline_topic_categories.py --list_sessions
```

#### Checkpoint Management:
- Checkpoints are automatically saved to the `checkpoints/` directory
- Each session gets a unique ID (e.g., `sync_20240728_143022`)
- Checkpoint files include:
  - `{session_id}_state.json`: Processing state and metadata
  - `{session_id}_items.pkl`: Remaining items to process
  - `{session_id}_results.json`: Intermediate results for recovery
- Automatic cleanup of checkpoint files after successful completion

### Batch Mode

Batch mode uses the OpenAI Batch API for cost-effective processing of large datasets:
- Submit all items as a single batch job
- Monitor batch status until completion
- Retrieve results when the batch is finished
- More cost-effective for large volumes but less real-time feedback

#### Usage Example:
```bash
python nde_pipeline_topic_categories.py --mode batch --dataset_path data.json --dataset_name zenodo
```

## Command Line Options

### Core Options:
- `--mode {sync,batch}`: Choose processing mode (default: batch)
- `--dataset_path`: Path to your dataset file
- `--dataset_name`: Dataset configuration name from column mappings
- `--output`: Custom output filename

### Sync Mode & Checkpointing:
- `--enable_checkpointing` / `--disable_checkpointing`: Control checkpointing (enabled by default)
- `--checkpoint_interval N`: Save checkpoint every N records (default: 10)
- `--session_id`: Custom session ID for checkpointing
- `--resume`: Resume from checkpoint using session_id
- `--list_sessions`: List all available checkpoint sessions

### Batch Processing:
- `--batch_ids`: List of existing batch IDs to process
- `--batch_file`: File containing batch IDs (one per line)
- `--check_interval`: Minutes between batch status checks (default: 30)

## Quick Start

### Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

### Basic Usage

**Sync Mode** (recommended for most use cases):
```bash
python nde_pipeline_topic_categories.py --mode sync --dataset_path data.json --dataset_name zenodo --output results.json
```

**Batch Mode** (for cost optimization):
```bash
python nde_pipeline_topic_categories.py --mode batch --dataset_path data.json --dataset_name zenodo --output results.json
```

## Data Preparation

Your dataset should be in JSON or CSV format. The pipeline uses column mappings to understand your data structure.

### Column Mappings Configuration

The column mappings are defined in `pipeline/utils/configs/column_mappings.json`. This file specifies how to extract information from your dataset:

- **text_columns**: Columns containing the text to be processed
- **id_column**: Column to use as unique identifier (auto-generated if not specified)
- **metadata_mapping**: Maps your dataset fields to expected field names

Example configuration:
```json
{
  "zenodo": {
    "text_columns": ["Description"],
    "id_column": "_id",
    "metadata_mapping": {
      "Name": "title",
      "URL": "source_url"
    }
  }
}
```

### Dataset Example

Your JSON dataset should look like this:
```json
[
  {
    "_id": "study_001",
    "Name": "Sample Study",
    "Description": "This study investigates...",
    "URL": "http://example.com"
  }
]
```

The pipeline will:
- Use `_id` as the unique identifier
- Process the `Description` field
- Map `Name` → `title` and `URL` → `source_url`

## How the Pipeline Works

The pipeline is designed to be modular and flexible, allowing for various components to work together seamlessly. Here's a general overview of how the pipeline operates:

1. **Input Handling**: The pipeline reads data from the specified dataset file using input classes that inherit from the `DataInput` base class. This allows for different file formats (e.g., CSV, JSON) to be processed uniformly.

2. **Preprocessing**: The data is cleaned and formatted using processors that inherit from the `DataProcessor` base class. These processors can be customized to perform various text cleaning and formatting tasks before the data is sent for categorization.

3. **Categorization**: The pipeline utilizes a language model provider that inherits from the `LLMProvider` base class. This component is responsible for interacting with the language model locally or via API, sending the cleaned prompts, and receiving the categorized results.

4. **Postprocessing**: The results are processed using additional processors that also inherit from the `DataProcessor` base class. This allows for normalization and other transformations to ensure that the output categories conform to expected formats.

5. **Output Handling**: Finally, the results are written to a specified output format using output classes that inherit from the `DataOutput` base class. This ensures that the processed results are saved in a structured and accessible manner.

By utilizing base classes for input, processing, categorization, and output, the pipeline maintains a high level of flexibility and extensibility, allowing users to customize and adapt the components to their specific needs.

## Advanced Usage

### Example Workflows

**Processing a large research dataset with checkpointing**:
```bash
# Start processing with conservative checkpoint interval
python nde_pipeline_topic_categories.py --mode sync \
  --dataset_path research_papers.json \
  --dataset_name zenodo \
  --checkpoint_interval 200 \
  --output results/research_categorized.json

# If interrupted, resume from checkpoint
python nde_pipeline_topic_categories.py --mode sync \
  --resume --session_id sync_20240728_143022
```

**Quick processing for testing**:
```bash
# Use batch mode for quick, small datasets
python nde_pipeline_topic_categories.py --mode batch \
  --dataset_path test_data.json \
  --dataset_name zenodo \
  --output results/test_results.json
```

### Best Practices

#### Choosing the Right Processing Mode

**Use Sync Mode when**:
- Processing large datasets that may take hours or days
- You need real-time progress feedback and results
- Reliability and crash recovery are important
- You want to process and save results incrementally
- You're running on unreliable infrastructure

**Use Batch Mode when**:
- You have a stable environment and processing time is predictable
- Cost optimization is a primary concern (batch API is more cost-effective)
- You can wait for all results at once
- The dataset size is manageable (completes within reasonable time)

#### Checkpoint Interval Guidelines

- **Small datasets (< 1000 records)**: Use `--checkpoint_interval 50-100`
- **Medium datasets (1000-10000 records)**: Use `--checkpoint_interval 100-500`  
- **Large datasets (> 10000 records)**: Use `--checkpoint_interval 500-1000`

Smaller intervals provide better crash recovery but create more I/O overhead. Larger intervals are more efficient but mean losing more progress if interrupted.

#### Recovery and Monitoring

**Monitor progress** during long-running jobs:
```bash
# Check checkpoint sessions
python nde_pipeline_topic_categories.py --list_sessions

# Resume if needed
python nde_pipeline_topic_categories.py --mode sync --resume --session_id <session_id>
```

**Automatic cleanup**: Checkpoint files are automatically cleaned up after successful completion. To keep them for debugging, use a custom checkpointer configuration.