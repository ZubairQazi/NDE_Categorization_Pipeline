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

async def main(dataset_path: str, dataset_name: str):
    # Load configurations
    config = Config()
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
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("edam_topics.txt")
    formatted_template = template_handler.format_template(template, topics)

    # Initialize processors
    processors = [
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
        Normalizer()
    ]
    
    # Initialize output handler
    output_handler = JSONOutput(
        output_dir=Path("results"),
        filename=f"{dataset_name}_results.json"
    )
    
    # Create and run pipeline
    pipeline = Pipeline(
        input_handler=input_handler,
        llm_provider=llm_provider,
        output_handler=output_handler,
        processors=processors,
        batch_size=10,
        max_retries=3,
        mode="batch"  # or "sync" for synchronous processing
    )
    
    # Run pipeline
    await pipeline.run()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <dataset_path> <dataset_name>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    asyncio.run(main(dataset_path, dataset_name)) 