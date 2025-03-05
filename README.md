# Topic Categorization Pipeline

## Overview

This project is a data processing pipeline designed to categorize scientific datasets using large language models. It includes components for data input, processing, and output, allowing users to efficiently manage and analyze their data.

## Initial Setup

To set up the project, follow these steps:

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
   Use the following command to install the project in editable mode:
   ```bash
   pip install -e .
   ```

   This command installs all required dependencies specified in the `pyproject.toml` file.

## Running the Pipeline Example

The pipeline example demonstrates how to run the data processing pipeline with a dataset. Here's how to use it:

1. **Prepare Your Dataset**:
   Ensure your dataset is in either CSV or JSON format. The dataset should contain the necessary fields as specified in your configuration. For more information on how the pipeline handles input data, refer to the [Input Handling](#input-handling) section. 

2. **Run the Pipeline Example**:
   You can run the pipeline example script using the command line. Here's the basic usage:
   ```bash
   python examples/pipeline_example.py --dataset_path <path_to_your_dataset> --dataset_name <your_dataset_name>
   ```

   You can also specify an output filename:
   ```bash
   python examples/pipeline_example.py --dataset_path <path_to_your_dataset> --dataset_name <your_dataset_name> --output <custom_output_name>
   ```

3. **Batch Processing**:
   If you want to process existing/completed batch IDs, you can do so by providing the batch IDs directly or from a file:
   ```bash
   python examples/pipeline_example.py --batch_ids <batch_id1> <batch_id2> --output <custom_output_name>
   ```
   or
   ```bash
   python examples/pipeline_example.py --batch_file <path_to_batch_ids_file> --output <custom_output_name>
   ```

### How the Pipeline Works

The pipeline is designed to be modular and flexible, allowing for various components to work together seamlessly. Here's a general overview of how the pipeline operates:

1. **Input Handling**: The pipeline reads data from the specified dataset file using input classes that inherit from the `DataInput` base class. This allows for different file formats (e.g., CSV, JSON) to be processed uniformly.

2. **Preprocessing**: The data is cleaned and formatted using processors that inherit from the `DataProcessor` base class. These processors can be customized to perform various text cleaning and formatting tasks before the data is sent for categorization.

3. **Categorization**: The pipeline utilizes a language model provider that inherits from the `LLMProvider` base class. This component is responsible for interacting with the language model locally or via API, sending the cleaned prompts, and receiving the categorized results.

4. **Postprocessing**: The results are processed using additional processors that also inherit from the `DataProcessor` base class. This allows for normalization and other transformations to ensure that the output categories conform to expected formats.

5. **Output Handling**: Finally, the results are written to a specified output format using output classes that inherit from the `DataOutput` base class. This ensures that the processed results are saved in a structured and accessible manner.

By utilizing base classes for input, processing, categorization, and output, the pipeline maintains a high level of flexibility and extensibility, allowing users to customize and adapt the components to their specific needs.

### Example of Batch Processing

In the `batch_processing_example.py`, the script demonstrates how to load a dataset, format prompts, and submit them for batch processing. Here's a brief overview of the steps:

1. **Load Configurations**: The script loads the necessary configurations and mappings for the dataset.
2. **Check Existing Batches**: It checks if there are any existing batches to process.
3. **Load Data**: The script loads the dataset using either `CSVInput` or `JSONInput`.
4. **Prepare Prompts**: It formats the prompts for each item based on the loaded template.
5. **Submit Batches**: The formatted prompts are submitted to the OpenAI API for processing.
6. **Monitor Results**: The script monitors the batch processing and logs the results.

## Input Handling

The pipeline reads data from the specified dataset file using either `CSVInput` or `JSONInput`, depending on the file format. You can also implement your own class for a different file format using `DataInput`. The input handling is designed to work with specific column mappings that define how to extract relevant information from your dataset.

### Column Mappings Configuration

The column mappings configuration is defined in a JSON file located at `pipeline/utils/configs/column_mappings.json`. This file specifies the structure of your dataset and how to map its fields to the expected input for the pipeline.

#### Structure of the Column Mappings JSON

The JSON file contains mappings for different datasets. Each dataset configuration includes:

- **text_columns**: A list of columns that contain the primary text data to be processed.
- **metadata_mapping**: A dictionary that maps the metadata fields in your dataset to the expected field names used in the pipeline. Expected field names are dependant upon your specific implementation.

Here's an example of the column mappings configuration:

```json
{
  "zenodo": {
    "text_columns": ["Description"],
    "metadata_mapping": {
      "Name": "title",
      "URL": "source_url",
      "Date": "date",
    }
  },
  "dbGaP": {
    "text_columns": ["description"],
    "metadata_mapping": {
      "name": "title",
      "url": "source_url"
    }
  }
}
```

### Using JSON Input with Column Mappings

1. **Prepare Your JSON Dataset**: Ensure your JSON dataset contains the necessary fields as specified in your column mappings configuration. The fields should match the keys defined in the `metadata_mapping`.

2. **Run the Pipeline Example**: You can run the pipeline example script using the command line. Here's how to specify the dataset and its configuration (matched via dataset name):
   ```bash
   python examples/pipeline_example.py --dataset_path <path_to_your_dataset.json> --dataset_name <your_dataset_name>
   ```

3. **How It Works**: When you run the pipeline with a JSON input, the `JSONInput` class will read the dataset and use the column mappings to extract the relevant text and metadata. The text data will be processed according to the specified `text_columns`, and the metadata will be mapped to the expected fields defined in `metadata_mapping`.

### Example of Column Mappings in Action

For instance, if your JSON dataset has the following structure:

```json
[
  {
    "Name": "Sample Study",
    "Description": "This study investigates...",
    "URL": "http://example.com"
  },
  {
    "Name": "Another Study",
    "Description": "This study explores...",
    "URL": "http://example.org"
  }
]
```

The pipeline will extract the `Description` field as the text to be processed and map the `Name` and `URL` fields to `title` and `source_url`, respectively, based on the column mappings configuration.