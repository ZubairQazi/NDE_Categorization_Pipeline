import asyncio
from pathlib import Path
import json
from datetime import datetime
import torch
import logging
import argparse
import sys
from typing import List

from pipeline import (
    Config,
    ColumnMappingsConfig,
    Pipeline,
    CSVInput,
    JSONInput,
    TextCleaner,
    Normalizer,
    JSONOutput,
    TemplateHandler,
    PromptFormatter
)
from pipeline.llm.hf_local_provider import HFLocalProvider
from pipeline.utils.logging import get_logger

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = get_logger(__name__)

def read_batch_ids_from_file(file_path: str) -> List[str]:
    """Read batch IDs from a file, one per line"""
    try:
        with open(file_path, 'r') as f:
            # Read lines and strip whitespace, filter out empty lines
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading batch IDs from file {file_path}: {e}")
        return []

async def main(
    dataset_path: str = None, 
    dataset_name: str = None, 
    model_name_or_path: str = None,
    output_filename: str = None,
    batch_size: int = 10,
    device: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
):
    logger.debug("Starting pipeline processing with HuggingFace local model...")
    
    # Load configurations
    config = Config()
    column_mappings = ColumnMappingsConfig()
    
    # For normal processing, we need dataset_path and dataset_name
    if not dataset_path or not dataset_name:
        raise ValueError("Dataset path and name are required for processing")
    
    dataset_config = column_mappings.get_dataset_config(dataset_name)
    
    # Initialize input handler based on file extension
    if dataset_path.lower().endswith('.csv'):
        input_handler = CSVInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"],
            id_column=dataset_config.get("id_column")
        )
    elif dataset_path.lower().endswith('.json'):
        input_handler = JSONInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"],
            id_column=dataset_config.get("id_column")
        )
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

    # Load template and configure prompt formatting
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("edam_topics.txt")
    formatted_template = template_handler.format_template(template, topics)
    
    # Initialize HuggingFace local provider
    llm_provider = HFLocalProvider(
        model_name_or_path=model_name_or_path or "mistralai/Mistral-7B-Instruct-v0.2",
        prompt_template=formatted_template,
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        temperature=0.1,
        max_new_tokens=512,
        max_retries=3,
        retry_delay=5
    )
    
    # Initialize processors
    preprocessors = [
        TextCleaner(
            remove_urls=True,
            remove_html=True,
            normalize_whitespace=True,
            max_length=None
        ),
        PromptFormatter(
            template=formatted_template,
            field_mappings={
                "title": "metadata.title",
                "abstract": "text"
            }
        ),
    ]

    postprocessors = [
        Normalizer(
            data_path="edam/EDAM.csv",
            edam_topics_path="edam/edam_topics.txt"
        )
    ]
    
    # Initialize output handler with custom filename if provided
    output_filename = output_filename or f"{dataset_name}_results.json"
    output_handler = JSONOutput(
        output_dir=Path("results"),
        filename=output_filename
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        input_handler=input_handler,
        llm_provider=llm_provider,
        output_handler=output_handler,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        categories=topics,
        batch_size=batch_size,
        max_retries=3,
        mode="sync"  # Local models work better in sync mode
    )
    
    # Run pipeline
    await pipeline.run()
    logger.info(f"Processing complete. Results saved to results/{output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data processing pipeline with a local HuggingFace model")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset file")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset configuration matched to column mappings")
    
    # Model arguments
    parser.add_argument("--model", help="Name or path of the HuggingFace model to use")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of items to process in each batch")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    
    # Output arguments
    parser.add_argument("--output", help="Custom output file name (without extension)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.load_in_8bit and args.load_in_4bit:
        print("Error: Cannot use both 8-bit and 4-bit quantization")
        sys.exit(1)
    
    # Run the pipeline
    output_filename = f"{args.output}_results.json" if args.output else f"{args.dataset_name}_results.json"
    
    asyncio.run(main(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        model_name_or_path=args.model,
        output_filename=output_filename,
        batch_size=args.batch_size,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    ))