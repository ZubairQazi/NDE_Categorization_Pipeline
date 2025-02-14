import glob
import json
import logging
import os
import sys
from tqdm import tqdm
import pandas as pd
import tiktoken
from openai import OpenAI

BATCH_INFO_FILE = "batch_info.json" 

class ConfigLoader:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            return json.load(f)


class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """Loads the dataset (CSV/JSON) and returns a pandas DataFrame."""
        if self.dataset_path.lower().endswith(".json"):
            return pd.read_json(self.dataset_path, lines=True if self.dataset_path.endswith(".jsonl") else False)
        elif self.dataset_path.lower().endswith(".csv"):
            return pd.read_csv(self.dataset_path, lineterminator="\n")
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")


class OpenAIBatchProcessor:
    def __init__(self, openai_api_key, openai_org_id, openai_project_id, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key, organization=openai_org_id, project=openai_project_id)
        self.model = model
        self.encoding_model = tiktoken.encoding_for_model(self.model)
        self.max_tokens_per_minute = 150000000
        self.max_requests_per_minute = 30000
        self.max_context_length = 128000

    def build_prompts(self, dataset, template_path="templates/prompt_template.txt", edam_file_path="EDAM/edam_topics.txt"):
        """Builds prompts for the dataset using a template."""
        # Load the prompt template and EDAM topics
        with open(template_path, "r") as template_file:
            template = template_file.read()
        with open(edam_file_path, "r") as edam_file:
            edam_topics = [topic.strip() for topic in edam_file.readlines()]

        formatted_topics = "\n".join(edam_topics)
        if "<topics>" in template:
            template = template.replace("<topics>", formatted_topics)

        # Prepare dataset columns
        dataset["description"] = dataset["description"].fillna("").astype(str)
        dataset["name"] = dataset["name"].fillna("").astype(str)
        dataset["_id"] = dataset["_id"].fillna("").astype(str)

        # Encoding the template without placeholders
        template_tokens = self.encoding_model.encode(template.replace("<title>", "").replace("<abstract>", "").replace("<num_terms>", ""))
        template_length = len(template_tokens)

        # Construct the prompts
        requests = []
        for idx, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            text = row["description"] if row["description"] != "" else "No Description"
            title = row["name"] if row["name"] != "" else "No Title"
            id = row["_id"] if row["_id"] != "" else "No ID"

            text_tokens = self.encoding_model.encode(text)
            title_tokens = self.encoding_model.encode(title)
            total_length = template_length + len(text_tokens) + len(title_tokens)

            # Ensure prompt doesn't exceed token limit
            while total_length > self.max_context_length:
                excess_tokens = total_length - self.max_context_length
                logging.warning(f"Prompt at index {idx} exceeds max length. Truncating by {excess_tokens} tokens.")
                text_tokens = text_tokens[:len(text_tokens) - excess_tokens]
                text = self.encoding_model.decode(text_tokens)
                prompt = template.replace("<abstract>", text).replace("<title>", title).replace("<num_terms>", "3")
                total_length = len(self.encoding_model.encode(prompt))

            # If the length is acceptable, build the full prompt
            if total_length <= self.max_context_length:
                prompt = template.replace("<abstract>", text).replace("<title>", title).replace("<num_terms>", "3")

            request = {
                "custom_id": f"{id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 248,
                },
            }

            requests.append(request)
        return requests

    def chunk_requests(self, requests, chunk_size=50000):
        """Chunks the requests into smaller batches."""
        for i in range(0, len(requests), chunk_size):
            yield requests[i:i + chunk_size]

    def create_batches(self, batch_files):
        """Creates batches on OpenAI and processes them, stores batch info."""
        batch_info_list = []
        for batch_file in batch_files:
            batch_input_file = self.client.files.create(file=open(batch_file, "rb"), purpose="batch")
            batch_input_file_id = batch_input_file.id

            # Create batch job on OpenAI
            suffix = os.path.splitext(batch_file)[0].split("_")[-1]
            batch_info = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"Batch {suffix}"},
            )
            logging.info(f"Created new batch: {batch_info.id}")

            # Store batch info
            batch_info_list.append({
                "batch_id": batch_info.id,
                "batch_file": batch_file,
                "input_file_id": batch_input_file_id
            })

        # Save batch information to a file for later use in formatting script
        with open(BATCH_INFO_FILE, "w") as f:
            json.dump(batch_info_list, f, indent=4)


class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        if self.dataset_path.lower().endswith(".json"):
            return pd.read_json(self.dataset_path, lines=True if self.dataset_path.endswith(".jsonl") else False)
        elif self.dataset_path.lower().endswith(".csv"):
            return pd.read_csv(self.dataset_path, lineterminator="\n")
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")


def main():
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.config
    openai_api_key = config["api_keys"]["openai"]
    openai_org_id = config["project_org_ids"]["openai_org"]
    openai_project_id = config["project_org_ids"]["openai_project"]

    # Initialize OpenAI processor
    processor = OpenAIBatchProcessor(openai_api_key, openai_org_id, openai_project_id)

    # Load dataset
    dataset_path = input("Enter dataset path (CSV or JSON): ")
    dataset_loader = DatasetLoader(dataset_path)
    dataset = dataset_loader.dataset

    # Build prompts and handle batching
    requests = processor.build_prompts(dataset)
    chunked_requests = processor.chunk_requests(requests)

    # Process batches
    processor.create_batches(chunked_requests)