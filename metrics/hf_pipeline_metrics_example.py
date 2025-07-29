import asyncio
from datetime import datetime
import logging
from pathlib import Path
import argparse
import sys

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
    PromptFormatter,
    HFLocalProvider
)
from pipeline.utils.logging import get_logger
from metrics.hf_collector import HuggingFaceMetricsCollector
from metrics.hf_metrics_wrapper import HFMetricsWrapper
from metrics.resource_monitor import ResourceMonitor
from pipeline.llm.hf_provider import HFProvider

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

async def run_pipeline_with_metrics(
    dataset_path: str = None,
    dataset_name: str = None,
    output_filename: str = None,
    model: str = "microsoft/Phi-4-mini-instruct",
    is_local: bool = True,
    local_model_path: str = None,
    mode: str = "sync"
):
    logger.info("Starting pipeline with HuggingFace metrics collection...")
    logger.info(f"Model being used: {local_model_path or model}, Mode: {mode}, Local: {is_local}")

    metrics_collector = HuggingFaceMetricsCollector(model=model, is_local=is_local)
    resource_monitor = ResourceMonitor(interval=2.0)  # Monitor every 2 seconds
    
    try:
        metrics_collector.start_collection()
        
        # Start resource monitoring
        def resource_callback(gpu_memory=None, cpu_usage=None, **kwargs):
            metrics_collector.record_resource_usage(
                gpu_memory=gpu_memory, cpu_usage=cpu_usage
            )
        
        resource_monitor.start(resource_callback)

        config = Config()
        template_handler = TemplateHandler()
        topics = template_handler.load_topics("edam_topics.txt")

        if not dataset_path or not dataset_name:
            raise ValueError("Dataset path and name are required for processing")

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

        # Load template and configure prompt formatting
        template = template_handler.load_template("prompt_template.txt")
        formatted_template = template_handler.format_template(template, topics)

        # Create HuggingFace provider
        if is_local:
            model_path = local_model_path or model
            base_provider = HFLocalProvider(
                model_name_or_path=model_path,
                prompt_template=formatted_template,
                load_in_4bit=True,  # Changed to 4-bit for lower memory usage
                load_in_8bit=False  # Explicitly disable 8-bit since we're using 4-bit
            )
            # Wrap with metrics collection
            llm_provider = HFMetricsWrapper(base_provider, metrics_collector)
        else:
            # For HuggingFace API, use the token from config
            hf_api_token = config.hf_api_token
            base_provider = HFProvider(
                api_token=hf_api_token,
                model_name=model,
                prompt_template=formatted_template,
                max_retries=3,
                retry_delay=5
            )
            # Wrap with metrics collection
            llm_provider = HFMetricsWrapper(base_provider, metrics_collector)

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

        output_filename = output_filename or f"{dataset_name}_results.json"
        output_handler = JSONOutput(
            output_dir=Path("results"),
            filename=output_filename
        )

        pipeline = Pipeline(
            input_handler=input_handler,
            llm_provider=llm_provider,  # Use the wrapped provider
            output_handler=output_handler,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            categories=topics,
            batch_size=50000,
            max_retries=3,
            mode=mode
        )

        results = await pipeline.run()

        # Stop resource monitoring
        resource_monitor.stop()
        
        metrics_collector.stop_collection()
        metrics_file = metrics_collector.save_metrics(f"{output_filename}_metrics.json")
        logger.info(f"Metrics saved to {metrics_file}")

        return results

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        resource_monitor.stop()
        metrics_collector.stop_collection()
        raise

async def main():
    parser = argparse.ArgumentParser(description="Run pipeline with HuggingFace metrics collection")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--output", type=str, help="Output filename prefix")
    parser.add_argument("--model", type=str, default="microsoft/Phi-4-mini-instruct", help="HuggingFace model to use")
    parser.add_argument("--local", action="store_true", help="Use local model")
    parser.add_argument("--local_model_path", type=str, help="Path to local model directory (if different from model name)")
    parser.add_argument("--mode", type=str, default="sync", choices=["sync", "batch"], help="Processing mode")
    args = parser.parse_args()

    if not args.dataset_path or not args.dataset_name:
        print("Usage:")
        print("  python hf_pipeline_metrics_example.py --dataset_path <path> --dataset_name <name> [--output <name>] [--model <model>] [--local]")
        sys.exit(1)

    output_filename = f"{args.output}_results.json" if args.output else f"{args.dataset_name}_results.json"
    await run_pipeline_with_metrics(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output_filename=output_filename,
        model=args.model,
        is_local=args.local,
        local_model_path=args.local_model_path,
        mode=args.mode
    )

if __name__ == "__main__":
    asyncio.run(main())