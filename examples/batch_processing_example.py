import asyncio
from pathlib import Path
import json
from datetime import datetime
from pipeline import (
    Config, 
    ColumnMappingsConfig,
    TemplateHandler,
    CSVInput,
    JSONInput,
    OpenAIProvider,
    TextItem,
)

async def main(dataset_path: str, dataset_name: str):
    # Load configurations
    config = Config()
    column_mappings = ColumnMappingsConfig()
    
    # Get dataset-specific mappings
    dataset_config = column_mappings.get_dataset_config(dataset_name)
    
    # Initialize OpenAI provider
    provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id
    )
    
    # Load template and topics
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("edam_topics.txt")
    formatted_template = template_handler.format_template(template, topics)     # Format template with topics
    
    # Check for existing batches
    can_proceed = await provider.check_existing_batches()
    if not can_proceed:
        print("Existing batches found. Please check and try again.")
        return

    # Load data with mappings from config
    if dataset_path.lower().endswith('.csv'):
        items = CSVInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"]
        ).get_text_items()
    elif dataset_path.lower().endswith('.json'):
        items = JSONInput(
            filepath=dataset_path,
            text_columns=dataset_config["text_columns"],
            metadata_mapping=dataset_config["metadata_mapping"]
        ).get_text_items()
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    
    # Take random length 10 subset
    import random
    if len(items) > 10:
        items = random.sample(items, 10)
    
    # Prepare prompts for each item
    ids, prompts = [], []
    for item in items:
        # Format the prompt template with item-specific data
        prompt = formatted_template.replace("<title>", item.metadata.get("title", "No Title"))
        prompt = prompt.replace("<abstract>", item.text)
        
        ids.append(item.id)
        prompts.append(prompt)
    
    # Submit batch with formatted prompts
    batch_ids = await provider.submit_batch(ids, prompts, f"batch_{dataset_name}")
    
    # Set up dataset-specific logging
    log_file = Path(f"logs/{dataset_name}_processing.log")
    log_file.parent.mkdir(exist_ok=True)
    
    # Monitor results
    results = []
    while batch_ids:
        for batch_id in batch_ids[:]:
            batch_results = await provider.get_batch_results(batch_id)
            if batch_results is not None:
                results.extend(batch_results)
                batch_ids.remove(batch_id)
                # Log progress
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now()}: Completed batch {batch_id}\n")
        if batch_ids:
            await asyncio.sleep(60 * 60)  # Every hour
    
    # Save results with dataset-specific name
    output_file = Path(f"results/{dataset_name}_results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <dataset_path> <dataset_name>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    asyncio.run(main(dataset_path, dataset_name))

    # Use tmux for multiple runs