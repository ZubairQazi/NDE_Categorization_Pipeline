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
    
    # Get job name from command line or default to test_job
    job_name = sys.argv[1] if len(sys.argv) > 1 else "test_job"
    
    # Initialize OpenAI provider with a per-job database
    provider = OpenAIProvider(
        api_key=config.openai_api_key,
        org_id=config.openai_org_id,
        project_id=config.openai_project_id,
        storage_dir="test_storage",
        job_name=job_name
    )
    
    # Load template and topics
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("topics/edam_topics.txt")
    
    # Format template with topics
    formatted_template = template_handler.format_template(template, topics)
    
    # Load test data
    items = CSVInput('pipeline/tests/data/zenodo.csv', text_columns=["Name", "Description"]).get_text_items()
    
    # Submit batch with the specified job name
    batch_ids = await provider.submit_batch(items, formatted_template, job_name)
    
    # Monitor job progress
    while True:
        status = provider.get_batch_info()
        completed = sum(1 for b in status if b["status"] == "completed")
        total_batches = len(status)
        print(f"Progress: {completed}/{total_batches} batches completed")
        
        if all(b["status"] in ["completed", "failed"] for b in status):
            break
        
        await asyncio.sleep(60 * 60 * 6)
    
    # Collect results
    results = []
    for batch in status:
        if batch['status'] == 'completed':
            batch_results = await provider.get_batch_results(batch['batch_id'])
            if batch_results:
                results.extend(batch_results)
    
    print("Test completed. Total results:", len(results))
    return results

if __name__ == "__main__":
    asyncio.run(main())
