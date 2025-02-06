import json
import logging
import sys

import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm

# Constants
MAX_TOKENS_PER_MINUTE = 1000000
MAX_REQUESTS_PER_MINUTE = 3500
MAX_CONTEXT_LENGTH = 16385
MODEL = "gpt-4o-mini"
BATCH_SIZE = 50000

# Logging configuration
logging.basicConfig(level=logging.INFO)


# Initialize the OpenAI client and configuration
def initialize_openai_client(config_path="config.json"):
    """Initialize the OpenAI client with API key and project ID from configuration."""
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    openai_api_key = config["api_keys"]["openai"]
    openai_org_id = config["project_org_ids"]["openai_org"]
    openai_project_id = config["project_org_ids"]["openai_project"]
    return OpenAI(
        api_key=openai_api_key, organization=openai_org_id, project=openai_project_id
    )


# Check and log existing batches
def check_existing_batches(client):
    """Check and log the status of existing batch jobs in OpenAI."""
    existing_batches = client.batches.list()
    if existing_batches:
        logging.info(f"Found {len(existing_batches.data)} existing batch jobs.")
        for batch in existing_batches.data:
            logging.info(f"Batch ID: {batch.id}, Status: {batch.status}")
            if batch.status == "failed":
                logging.error(f"Batch ID: {batch.id} failed with errors.")
            if batch.status == "running":
                logging.warning(
                    "Running batch job found. Exiting to prevent conflicts."
                )
                sys.exit()
    else:
        logging.info("No existing batch jobs found.")


# Load dataset
def load_dataset(dataset_path):
    """Load dataset from the specified path."""
    dataset = pd.read_csv(dataset_path, lineterminator="\n")
    logging.info(f"Loaded dataset with {len(dataset)} rows.")
    return dataset


# Get columns to process
def get_column_names(dataset):
    """Prompt the user for column names if necessary and verify their existence in the dataset."""
    columns = ["Name", "Description", "_id"]
    if not all(col in dataset.columns for col in columns):
        columns[0] = input(
            f"Enter the column for title ({', '.join(dataset.columns)}): "
        )
        columns[1] = input(
            f"Enter the column for description ({', '.join(dataset.columns)}): "
        )
        columns[2] = input(
            f"Enter the column for identifier ({', '.join(dataset.columns)}): "
        )
    return columns


# Generate prompt from dataset row
def build_prompt(template, row, encoding_model, text_col, title_col, id_col):
    """Generate prompt using the provided row data, truncating if necessary."""
    text, title, _id = row[text_col], row[title_col], row[id_col]
    text_tokens = encoding_model.encode(text)
    title_tokens = encoding_model.encode(title)
    total_length = (
        len(encoding_model.encode(template)) + len(text_tokens) + len(title_tokens)
    )

    if total_length > MAX_CONTEXT_LENGTH:
        excess_tokens = total_length - MAX_CONTEXT_LENGTH
        logging.warning(f"Truncating text by {excess_tokens} tokens.")
        text_tokens = text_tokens[: len(text_tokens) - excess_tokens]
        text = encoding_model.decode(text_tokens)

    prompt = (
        template.replace("<abstract>", text)
        .replace("<title>", title)
        .replace("<num_terms>", "3")
    )
    return prompt


# Build requests from dataset
def build_requests(dataset, template, encoding_model, text_col, title_col, id_col):
    """Construct requests from the dataset rows."""
    requests = []
    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        prompt = build_prompt(
            template, row, encoding_model, text_col, title_col, id_col
        )
        request = {
            "custom_id": str(row[id_col]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 248,
            },
        }
        requests.append(request)
    return requests


# Write JSONL batch files and create batches
def create_batches(client, requests, batch_size=BATCH_SIZE):
    """Chunk requests into batches, write JSONL files, and upload them to OpenAI."""
    for i, chunk in enumerate(
        tqdm(
            [requests[x : x + batch_size] for x in range(0, len(requests), batch_size)]
        )
    ):
        jsonl_data = "\n".join(json.dumps(req) for req in chunk)
        batch_file_path = f"datasets/batch_requests/batchinput_{i}.jsonl"
        with open(batch_file_path, "w") as file:
            file.write(jsonl_data)

        batch_input_file = client.files.create(
            file=open(batch_file_path, "rb"), purpose="batch"
        )
        client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"GPT Topics Batch {i}"},
        )


# Main execution
def main():
    client = initialize_openai_client()
    check_existing_batches(client)

    dataset_path = input("Enter dataset path (CSV): ")
    dataset = load_dataset(dataset_path)

    rebuild_jsonl = input("Do you want to rebuild the JSONL file? (y/n): ").lower()
    if rebuild_jsonl == "y":
        text_col, title_col, id_col = get_column_names(dataset)

        with open("templates/measurement_techniques.txt") as template_file:
            template = template_file.read()
        with open("EDAM/edam_topics.txt") as edam_file:
            template = template.replace(
                "<topics>", "\n".join(line.strip() for line in edam_file)
            )

        encoding_model = tiktoken.encoding_for_model(MODEL)
        requests = build_requests(
            dataset, template, encoding_model, text_col, title_col, id_col
        )
        create_batches(client, requests)


if __name__ == "__main__":
    main()
