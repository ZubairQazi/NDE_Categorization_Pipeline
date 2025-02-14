import asyncio
from pathlib import Path
from pipeline.utils.config import Config
from pipeline.utils.template import TemplateHandler
from pipeline.input.csv_input import CSVInput
from pipeline.llm.openai_provider import OpenAIProvider

async def main():
    # Load configuration
    config = Config("config.json")
    
    # Initialize OpenAI provider
    provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id
    )
    
    # Load template and topics
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("topics/edam_topics.txt")
    
    # Format template with topics
    formatted_template = template_handler.format_template(template, topics)
    
    # Check for existing batches
    can_proceed = await provider.check_existing_batches()
    if not can_proceed:
        print("Existing batches found. Please check and try again.")
        return

    items = CSVInput('pipeline/tests/data/zenodo.csv', text_columns=["Name", "Description"]).get_text_items()
    
    # Submit batch
    batch_ids = await provider.submit_batch(items, formatted_template, "test_batch")
    
    # Monitor results
    results = []
    while batch_ids:
        for batch_id in batch_ids[:]:
            batch_results = await provider.get_batch_results(batch_id)
            if batch_results is not None:
                results.extend(batch_results)
                batch_ids.remove(batch_id)
        if batch_ids:
            await asyncio.sleep(60)  # Check every minute
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 