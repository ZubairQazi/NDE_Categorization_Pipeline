import asyncio
from pathlib import Path
import json
from datetime import datetime
from pipeline import (
    Config,
    ColumnMappingsConfig,
    Pipeline,
    CSVInput,
    JSONInput,
    OpenAIProvider,
    TextCleaner,
    Normalizer,
    JSONOutput,
    TemplateHandler,
    PromptFormatter
)
import logging
from pipeline.utils.logging import get_logger
from typing import List
import argparse

# Set up logging at the start of your script
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

async def main(dataset_path: str = None, dataset_name: str = None, existing_batch_ids: List[str] = None, output_filename: str = None):
    logger.debug("Starting pipeline processing...")
    
    # Load configurations
    config = Config()
    
    template_handler = TemplateHandler()
    topics = template_handler.load_topics("edam_topics.txt")

    # If we're only processing existing batches, we can skip some initialization
    if existing_batch_ids:
        logger.info(f"Processing {len(existing_batch_ids)} existing batch IDs")
        
        # Initialize OpenAI provider
        llm_provider = OpenAIProvider(
            api_key=config.openai_api_key,
            org_id=config.openai_org_id,
            project_id=config.openai_project_id
        )
        
        # Initialize output handler with custom filename if provided
        output_filename = output_filename or "batch_results.json"
        output_handler = JSONOutput(
            output_dir=Path("results"),
            filename=output_filename
        )

        postprocessors = [
            Normalizer(
                data_path="edam/EDAM.csv",
                edam_topics_path="edam/edam_topics.txt"
            )
        ]
        
        # Create and run pipeline with existing batch IDs
        pipeline = Pipeline(
            input_handler=None,  # No input handler needed
            llm_provider=llm_provider,
            output_handler=output_handler,
            preprocessors=[],  # Only output processors are needed
            postprocessors=postprocessors,
            existing_batch_ids=existing_batch_ids,
            categories=topics,
        )
        
        # Run pipeline
        await pipeline.run()
        return
    
    # For normal processing, we need dataset_path and dataset_name
    if not dataset_path or not dataset_name:
        raise ValueError("Dataset path and name are required for normal processing")
    
    column_mappings = ColumnMappingsConfig()
    dataset_config = column_mappings.get_dataset_config(dataset_name)
    
    # Initialize components
    if dataset_path.lower().endswith('.csv'):
        input_handler = CSVInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"]
        )
    elif dataset_path.lower().endswith('.json'):
        input_handler = JSONInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"]
        )
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

    # Initialize OpenAI provider
    llm_provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id
    )
    
    # Load template and configure prompt formatting
    template = template_handler.load_template("prompt_template.txt")
    formatted_template = template_handler.format_template(template, topics)

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
        Normalizer()
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
        preprocessors=preprocessors ,
        postprocessors=postprocessors,
        categories=topics,
        batch_size=50000,
        max_retries=3,
        mode="batch"  # or "sync" for synchronous processing
    )
    
    # Run pipeline
    await pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")
    
    # Create a group for normal processing
    normal_group = parser.add_argument_group('Normal Processing')
    normal_group.add_argument("--dataset_path", help="Path to the dataset file")
    normal_group.add_argument("--dataset_name", help="Name of the dataset configuration matched to column mappings")
    
    # Create a mutually exclusive group for batch processing options
    batch_group = parser.add_argument_group('Batch Processing')
    batch_exclusive = batch_group.add_mutually_exclusive_group()
    batch_exclusive.add_argument("--batch_ids", nargs="+", help="List of existing batch IDs to process")
    batch_exclusive.add_argument("--batch_file", help="Path to file containing batch IDs (one per line)")
    
    # Add output file name argument
    parser.add_argument("--output", help="Custom output file name (without extension)")
    
    args = parser.parse_args()
    
    # Determine which mode to run in
    batch_mode = args.batch_ids is not None or args.batch_file is not None
    normal_mode = args.dataset_path is not None and args.dataset_name is not None
    
    if batch_mode and normal_mode:
        print("Error: Cannot specify both batch processing and normal processing options")
        sys.exit(1)
    
    if batch_mode:
        # Get batch IDs either directly or from file
        batch_ids = []
        if args.batch_ids:
            batch_ids = args.batch_ids
        elif args.batch_file:
            batch_ids = read_batch_ids_from_file(args.batch_file)
            
        if not batch_ids:
            print("Error: No valid batch IDs provided")
            sys.exit(1)
            
        # Process existing batches only
        # Use custom output name if provided
        output_filename = f"{args.output}_results.json" if args.output else "batch_results.json"
        asyncio.run(main(existing_batch_ids=batch_ids, output_filename=output_filename))
    elif normal_mode:
        # Normal pipeline processing
        # Use custom output name if provided
        output_filename = f"{args.output}_results.json" if args.output else f"{args.dataset_name}_results.json"
        asyncio.run(main(args.dataset_path, args.dataset_name, output_filename=output_filename))
    else:
        import sys
        print("Usage for normal processing:")
        print("  python script.py --dataset_path <path> --dataset_name <name> [--output <name>]")
        print("\nUsage for batch processing:")
        print("  python script.py --batch_ids <batch_id1> <batch_id2> ... [--output <name>]")
        print("  OR")
        print("  python script.py --batch_file <path_to_batch_ids_file> [--output <name>]")
        sys.exit(1) 