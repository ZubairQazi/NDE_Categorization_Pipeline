import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from pipeline import (
    ColumnMappingsConfig,
    Config,
    CSVInput,
    JSONInput,
    JSONOutput,
    Normalizer,
    OpenAIProvider,
    Pipeline,
    PromptFormatter,
    SyncCheckpointer,
    TemplateHandler,
    TextCleaner,
)
from pipeline.utils.batch_monitor import BatchMonitor
from pipeline.utils.logging import get_logger

# Set up logging at the start of your script - filter out noisy debug logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress noisy third-party loggers
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = get_logger(__name__)


def read_batch_ids_from_file(file_path: str) -> List[str]:
    """Read batch IDs from a file, one per line"""
    try:
        with open(file_path, "r") as f:
            # Read lines and strip whitespace, filter out empty lines
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading batch IDs from file {file_path}: {e}")
        return []


async def process_existing_batches(
    batch_ids: List[str], output_filename: str = None, check_interval_minutes: int = 30
):
    """Process existing batch IDs with enhanced monitoring and error handling"""
    logger.info(f"Processing {len(batch_ids)} existing batch IDs")
    logger.info(f"Will check for completion every {check_interval_minutes} minutes")

    # Initialize OpenAI provider
    config = Config()
    llm_provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id,
    )

    # Initialize batch monitor
    monitor = BatchMonitor(llm_provider, check_interval_minutes)

    # Create progress logger
    progress_callback = monitor.create_progress_logger(log_to_file=True)

    # Wait for batch completion with monitoring
    logger.info("Starting batch monitoring - you can safely leave this running in tmux")
    batch_results = await monitor.wait_for_completion_with_monitoring(
        batch_ids, progress_callback
    )

    if not batch_results:
        logger.error("No results retrieved from any batch")
        return

    # Collect all results
    all_results = []
    for batch_id, results in batch_results.items():
        logger.info(f"Retrieved {len(results)} results from batch {batch_id}")
        all_results.extend(results)

    # Process results with normalizer
    if all_results:
        template_handler = TemplateHandler()
        topics = template_handler.load_topics("edam_topics.txt")

        postprocessors = [
            Normalizer(
                data_path="edam/EDAM.csv", edam_topics_path="edam/edam_topics.txt"
            )
        ]

        # Apply post-processing
        processed_results = all_results
        for processor in postprocessors:
            processed_results = processor.process_output(processed_results)

        # Save results
        output_filename = output_filename or "batch_results.json"
        output_handler = JSONOutput(
            output_dir=Path("results"), filename=output_filename
        )

        output_handler.write(processed_results)
        logger.info(
            f"Saved {len(processed_results)} processed results to {output_filename}"
        )

        # Print final summary
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total batches processed: {len(batch_results)}")
        print(f"Total results retrieved: {len(all_results)}")
        print(f"Results after post-processing: {len(processed_results)}")
        print(f"Output file: results/{output_filename}")
        print(f"Log files saved in: logs/")
    else:
        logger.warning("No results to save")


async def main(
    dataset_path: str = None,
    dataset_name: str = None,
    existing_batch_ids: List[str] = None,
    output_filename: str = None,
    check_interval_minutes: int = 30,
    # Sync mode options
    mode: str = "batch",
    enable_checkpointing: bool = True,
    checkpoint_interval: int = 10,
    session_id: str = None,
    resume_from_checkpoint: bool = False,
    list_sessions: bool = False,
):
    logger.debug("Starting pipeline processing...")

    # Handle checkpoint session listing
    if list_sessions:
        sessions = Pipeline.list_checkpoint_sessions()
        if sessions:
            print("Available checkpoint sessions:")
            print("=" * 40)
            for session in sessions:
                info = Pipeline.get_session_info(session)
                if info:
                    print(f"Session: {session}")
                    print(f"  Timestamp: {info.get('timestamp', 'Unknown')}")
                    print(f"  Processed: {info.get('processed_count', 0)} items")
                    print(f"  Batches: {info.get('batch_count', 0)}")
                    print(f"  Results: {info.get('total_results', 0)}")
                    if info.get("additional_state", {}).get("emergency_save"):
                        print(f"  Status: EMERGENCY SAVE (crashed)")
                    elif info.get("additional_state", {}).get("completed"):
                        print(f"  Status: COMPLETED")
                    else:
                        print(f"  Status: IN PROGRESS")
                    print()
        else:
            print("No checkpoint sessions found.")
        return

    # If we're only processing existing batches, use the dedicated function
    if existing_batch_ids:
        await process_existing_batches(
            existing_batch_ids, output_filename, check_interval_minutes
        )
        return

    # For normal processing, we need dataset_path and dataset_name
    if not dataset_path or not dataset_name:
        raise ValueError("Dataset path and name are required for normal processing")

    # Load configurations
    config = Config()
    template_handler = TemplateHandler()
    topics = template_handler.load_topics("edam_topics.txt")

    column_mappings = ColumnMappingsConfig()
    dataset_config = column_mappings.get_dataset_config(dataset_name)

    # Initialize components
    if dataset_path.lower().endswith(".csv"):
        input_handler = CSVInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"],
            id_column=dataset_config.get("id_column"),
        )
    elif dataset_path.lower().endswith(".json"):
        input_handler = JSONInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"],
            id_column=dataset_config.get("id_column"),
        )
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

    # Initialize OpenAI provider
    llm_provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id,
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
            max_length=None,
        ),
        PromptFormatter(
            template=formatted_template,
            field_mappings={"title": "metadata.title", "abstract": "text"},
        ),
    ]

    postprocessors = [
        Normalizer(data_path="edam/EDAM.csv", edam_topics_path="edam/edam_topics.txt")
    ]

    # Initialize output handler with custom filename if provided
    output_filename = output_filename or f"{dataset_name}_results.json"
    output_handler = JSONOutput(output_dir=Path("results"), filename=output_filename)

    # Create and run pipeline
    pipeline = Pipeline(
        input_handler=input_handler,
        llm_provider=llm_provider,
        output_handler=output_handler,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        categories=topics,
        batch_size=50000 if mode == "batch" else 10,  # Smaller batches for sync mode
        max_retries=3,
        mode=mode,
        # Sync mode specific options
        enable_checkpointing=enable_checkpointing,
        checkpoint_interval=checkpoint_interval,
        session_id=session_id,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # Show checkpoint status for sync mode
    if mode == "sync":
        status = pipeline.get_checkpoint_status()
        if status:
            logger.info(f"Sync mode checkpoint status: {status}")

    # Run pipeline
    await pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")

    # Create a group for normal processing
    normal_group = parser.add_argument_group("Normal Processing")
    normal_group.add_argument("--dataset_path", help="Path to the dataset file")
    normal_group.add_argument(
        "--dataset_name",
        help="Name of the dataset configuration matched to column mappings",
    )
    normal_group.add_argument(
        "--mode",
        choices=["sync", "batch"],
        default="batch",
        help="Processing mode: 'sync' for synchronous with checkpointing, 'batch' for OpenAI batch API (default: batch)",
    )

    # Create a mutually exclusive group for batch processing options
    batch_group = parser.add_argument_group("Batch Processing")
    batch_exclusive = batch_group.add_mutually_exclusive_group()
    batch_exclusive.add_argument(
        "--batch_ids", nargs="+", help="List of existing batch IDs to process"
    )
    batch_exclusive.add_argument(
        "--batch_file", help="Path to file containing batch IDs (one per line)"
    )

    # Add batch monitoring options
    batch_group.add_argument(
        "--check_interval",
        type=int,
        default=30,
        help="Minutes between batch status checks (default: 30)",
    )

    # Sync mode and checkpoint options
    sync_group = parser.add_argument_group("Sync Mode & Checkpointing")
    sync_group.add_argument(
        "--enable_checkpointing",
        action="store_true",
        default=True,
        help="Enable checkpointing for sync mode (default: True)",
    )
    sync_group.add_argument(
        "--disable_checkpointing",
        action="store_true",
        help="Disable checkpointing for sync mode",
    )
    sync_group.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save checkpoint every N records in sync mode (default: 10)",
    )
    sync_group.add_argument(
        "--session_id",
        help="Custom session ID for checkpointing (auto-generated if not provided)",
    )
    sync_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint using session_id",
    )
    sync_group.add_argument(
        "--list_sessions",
        action="store_true",
        help="List all available checkpoint sessions and exit",
    )

    # Add output file name argument
    parser.add_argument("--output", help="Custom output file name (without extension)")

    args = parser.parse_args()

    # Handle checkpoint session listing first
    if args.list_sessions:
        asyncio.run(main(list_sessions=True))
        sys.exit(0)

    # Determine which mode to run in
    batch_mode = args.batch_ids is not None or args.batch_file is not None
    normal_mode = args.dataset_path is not None and args.dataset_name is not None

    if batch_mode and normal_mode:
        print(
            "Error: Cannot specify both batch processing and normal processing options"
        )
        sys.exit(1)

    # Process checkpointing options
    enable_checkpointing = args.enable_checkpointing and not args.disable_checkpointing

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
        output_filename = (
            f"{args.output}_results.json" if args.output else "batch_results.json"
        )
        asyncio.run(
            main(
                existing_batch_ids=batch_ids,
                output_filename=output_filename,
                check_interval_minutes=args.check_interval,
            )
        )
    elif normal_mode:
        # Normal pipeline processing
        # Use custom output name if provided
        output_filename = (
            f"{args.output}_results.json"
            if args.output
            else f"{args.dataset_name}_results.json"
        )
        asyncio.run(
            main(
                dataset_path=args.dataset_path,
                dataset_name=args.dataset_name,
                output_filename=output_filename,
                mode=args.mode,
                enable_checkpointing=enable_checkpointing,
                checkpoint_interval=args.checkpoint_interval,
                session_id=args.session_id,
                resume_from_checkpoint=args.resume,
            )
        )
    else:
        print("NDE Pipeline - Enhanced Batch & Sync Processing")
        print("=" * 55)
        print("\nUsage for normal processing:")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --dataset_path <path> \\")
        print("    --dataset_name <name> \\")
        print("    [--mode {batch,sync}] \\")
        print("    [--output <name>]")
        print("\nUsage for batch monitoring (existing batches):")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --batch_ids <batch_id1> <batch_id2> ... \\")
        print("    [--output <name>] \\")
        print("    [--check_interval <minutes>]")
        print("  OR")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --batch_file <path_to_batch_ids_file> \\")
        print("    [--output <name>] \\")
        print("    [--check_interval <minutes>]")
        print("\nUsage for checkpoint management:")
        print("  python nde_pipeline_topic_categories.py --list_sessions")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --dataset_path <path> --dataset_name <name> \\")
        print("    --mode sync --resume --session_id <session_id>")
        print("\nFeatures:")
        print("  • BATCH MODE: OpenAI Batch API with intelligent monitoring")
        print("  • SYNC MODE: Real-time processing with checkpointing & recovery")
        print("  • Automatic progress tracking and ETA estimation")
        print("  • Crash recovery and resume functionality")
        print("  • Comprehensive logging and intermediate result saving")
        print("  • Post-processing with EDAM topic normalization")
        print("\nExamples:")
        print("  # Create new batch (async processing)")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --dataset_path pipeline/tests/data/mini_zenodo.csv \\")
        print("    --dataset_name zenodo --mode batch")
        print("\n  # Sync processing with checkpointing")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --dataset_path pipeline/tests/data/mini_zenodo.csv \\")
        print("    --dataset_name zenodo --mode sync")
        print("\n  # Resume from checkpoint after crash")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --dataset_path pipeline/tests/data/mini_zenodo.csv \\")
        print("    --dataset_name zenodo --mode sync \\")
        print("    --resume --session_id sync_20250128_143022")
        print("\n  # Monitor existing batch")
        print("  python nde_pipeline_topic_categories.py \\")
        print("    --batch_ids batch_12345 --check_interval 5")
        print("\n  # List checkpoint sessions")
        print("  python nde_pipeline_topic_categories.py --list_sessions")
        sys.exit(1)
