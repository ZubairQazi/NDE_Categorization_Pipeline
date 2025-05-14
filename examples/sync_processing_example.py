import asyncio
from pathlib import Path
import json
from datetime import datetime
import logging
import argparse
from typing import List

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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = get_logger(__name__)

async def main(
    dataset_path: str,
    dataset_name: str,
    output_filename: str = None,
):
    logger.debug("Starting synchronous pipeline processing with OpenAI...")
    
    # Load configurations
    config = Config()
    column_mappings = ColumnMappingsConfig()
    
    # Get dataset-specific mappings
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

    # Initialize OpenAI provider
    llm_provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id,
        model="gpt-4o-mini",  # You can change this to your preferred model
        max_tokens=2048  # Adjust based on your needs
    )
    
    # Load template and configure prompt formatting
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("edam_topics.txt")
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
    
    # Initialize output handler with custom filename if provided
    output_filename = output_filename or f"{dataset_name}_results.json"
    output_handler = JSONOutput(
        output_dir=Path("results"),
        filename=output_filename
    )
    
    # Create and run pipeline in sync mode
    pipeline = Pipeline(
        input_handler=input_handler,
        llm_provider=llm_provider,
        output_handler=output_handler,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        categories=topics,
        max_retries=3,
        mode="sync"  # Run in synchronous mode
    )
    
    # Run pipeline
    await pipeline.run()
    logger.info(f"Processing complete. Results saved to results/{output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline synchronously with OpenAI")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset file")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset configuration")
    parser.add_argument("--output", help="Custom output file name")
    
    args = parser.parse_args()
    
    # Run the pipeline
    asyncio.run(main(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output_filename=args.output,
    )) 