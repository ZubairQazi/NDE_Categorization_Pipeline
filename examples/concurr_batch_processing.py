import asyncio
import sys
from pathlib import Path
from pipeline.utils.config import Config
from pipeline.utils.template import TemplateHandler
from pipeline.input.csv_input import CSVInput
from pipeline.llm.concurr_openai_provider import OpenAIProvider

async def main():
    # Load configuration
    config = Config("config.json")
    
    # Initialize provider
    provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id,
        storage_dir="test_storage"
    )
    
    # Load template and topics
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("topics/edam_topics.txt")
    
    # Format template with topics
    formatted_template = template_handler.format_template(template, topics)
    
    # Load test data
    items = CSVInput('pipeline/tests/data/zenodo.csv', text_columns=["Name", "Description"]).get_text_items()
    
    # Submit batch and get results directly
    results = await provider.batch_categorize(
        ids=[item.id for item in items],
        prompts=[formatted_template.replace("<abstract>", item.text) for item in items],
        batch_name="test_batch"
    )
    
    print("Test completed. Total results:", len(results))
    return results

if __name__ == "__main__":
    asyncio.run(main())
