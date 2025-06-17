import asyncio
from datetime import datetime
import logging
from pathlib import Path
import json
import argparse
import sys

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
from pipeline.utils.logging import get_logger
from metrics import OpenAIMetricsCollector, ResourceMonitor
from metrics.llm_metrics_wrapper import LLMMetricsWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def read_batch_ids_from_file(file_path: str):
    """Read batch IDs from a file, one per line"""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading batch IDs from file {file_path}: {e}")
        return []

async def run_pipeline_with_metrics(
    dataset_path: str = None,
    dataset_name: str = None,
    existing_batch_ids: list = None,
    output_filename: str = None,
    model: str = "gpt-4.1-nano",
    mode: str = "sync"
):
    """Run the pipeline with metrics collection"""
    logger.info("Starting pipeline with metrics collection...")
    logger.info(f"Model being used: {model}, Mode: {mode}")
    
    # Initialize metrics collector
    metrics_collector = OpenAIMetricsCollector(model=model)
    resource_monitor = ResourceMonitor(interval=1.0)
    
    try:
        # Start metrics collection
        metrics_collector.start_collection()
        resource_monitor.start(metrics_collector.record_resource_usage)
        
        # Load configurations
        config = Config()
        template_handler = TemplateHandler()
        topics = template_handler.load_topics("edam_topics.txt")
        
        # If processing existing batches
        if existing_batch_ids:
            logger.info(f"Processing {len(existing_batch_ids)} existing batch IDs")
            
            # Create base provider and wrap it with metrics
            base_provider = OpenAIProvider(
                api_key=config.openai_api_key,
                org_id=config.openai_org_id,
                project_id=config.openai_project_id
            )
            llm_provider = LLMMetricsWrapper(base_provider, metrics_collector)
            
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
            
            pipeline = Pipeline(
                input_handler=None,
                llm_provider=llm_provider,
                output_handler=output_handler,
                preprocessors=[],
                postprocessors=postprocessors,
                existing_batch_ids=existing_batch_ids,
                categories=topics,
            )
            
            # Run pipeline
            results = await pipeline.run()
            
        else:
            # Normal processing
            if not dataset_path or not dataset_name:
                raise ValueError("Dataset path and name are required for normal processing")
            
            column_mappings = ColumnMappingsConfig()
            dataset_config = column_mappings.get_dataset_config(dataset_name)
            
            # Initialize input handler
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
            
            # Create base provider and wrap it with metrics
            base_provider = OpenAIProvider(
                api_key=config.openai_api_key,
                org_id=config.openai_org_id,
                project_id=config.openai_project_id
            )
            llm_provider = LLMMetricsWrapper(base_provider, metrics_collector)
            
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
                Normalizer(
                    data_path="edam/EDAM.csv",
                    edam_topics_path="edam/edam_topics.txt"
                )
            ]
            
            # Initialize output handler
            output_filename = output_filename or f"{dataset_name}_results.json"
            output_handler = JSONOutput(
                output_dir=Path("results"),
                filename=output_filename
            )
            
            # Create pipeline
            pipeline = Pipeline(
                input_handler=input_handler,
                llm_provider=llm_provider,
                output_handler=output_handler,
                preprocessors=preprocessors,
                postprocessors=postprocessors,
                categories=topics,
                batch_size=50000,
                max_retries=3,
                mode=mode
            )
            
            # Run pipeline
            results = await pipeline.run()
        
        # Stop metrics collection
        metrics_collector.stop_collection()
        resource_monitor.stop()
        
        # Save metrics
        metrics_file = metrics_collector.save_metrics(f"{output_filename}_metrics.json")
        logger.info(f"Metrics saved to {metrics_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        metrics_collector.stop_collection()
        resource_monitor.stop()
        raise

async def main():
    parser = argparse.ArgumentParser(description="Run pipeline with metrics collection")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--output", type=str, help="Output filename prefix")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--mode", type=str, default="sync", choices=["sync", "batch"], help="Processing mode")
    parser.add_argument("--batch_ids", nargs="+", help="List of batch IDs to process")
    parser.add_argument("--batch_file", type=str, help="Path to file containing batch IDs")
    args = parser.parse_args()

    # Determine which mode to run in
    has_batch_ids = args.batch_ids is not None or args.batch_file is not None
    has_dataset = args.dataset_path is not None and args.dataset_name is not None

    if has_batch_ids and has_dataset:
        print("Error: Cannot specify both batch processing and normal processing options")
        sys.exit(1)

    if has_batch_ids:
        # Get batch IDs either directly or from file
        batch_ids = []
        if args.batch_ids:
            batch_ids = args.batch_ids
        elif args.batch_file:
            batch_ids = read_batch_ids_from_file(args.batch_file)
        if not batch_ids:
            print("Error: No valid batch IDs provided")
            sys.exit(1)
        # Process existing batches
        output_filename = f"{args.output}_results.json" if args.output else "batch_results.json"
        await run_pipeline_with_metrics(
            existing_batch_ids=batch_ids,
            output_filename=output_filename,
            model=args.model,
            mode=args.mode
        )
    elif has_dataset:
        # Normal pipeline processing
        output_filename = f"{args.output}_results.json" if args.output else f"{args.dataset_name}_results.json"
        await run_pipeline_with_metrics(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            output_filename=output_filename,
            model=args.model,
            mode="sync"  # Force sync mode for dataset processing
        )
    else:
        print("Usage for normal processing:")
        print("  python script.py --dataset_path <path> --dataset_name <name> [--output <name>] [--model <model>]")
        print("\nUsage for batch processing:")
        print("  python script.py --batch_ids <batch_id1> <batch_id2> ... [--output <name>] [--model <model>] [--mode <sync|batch>]")
        print("  OR")
        print("  python script.py --batch_file <path_to_batch_ids_file> [--output <name>] [--model <model>] [--mode <sync|batch>]")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 