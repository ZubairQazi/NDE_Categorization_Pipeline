# llm/openai_provider.py

from typing import List, Optional, Dict
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import json
import tiktoken
from openai import OpenAI
from .base import LLMProvider
from ..core.data_model import TextItem, CategoryResult, BatchJob, JobStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)

class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 248,
        max_context_length: int = 128000,
        batch_chunk_size: int = 50000,
        completion_window: str = "24h"
    ):
        self.client = OpenAI(
            api_key=api_key,
            organization=org_id,
            project=project_id
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_context_length = max_context_length
        self.batch_chunk_size = batch_chunk_size
        self.completion_window = completion_window
        self.encoding = tiktoken.encoding_for_model(model)

    async def check_existing_batches(self) -> bool:
        """Check for existing batch jobs but do not block new batches."""
        existing_batches = self.client.batches.list()
        if existing_batches:
            running_batch_count = sum(1 for batch in existing_batches.data 
                                    if batch.status in ["in_progress", "finalizing", "validating", "cancelling"])
            if running_batch_count > 0:
                logger.warning(f"There are {running_batch_count} running batch jobs.")
        return True  # Always returns True, so it won't block

    def _create_batch_request(self, item_id: str, prompt: str) -> Dict:
        """Create a single batch request for an item"""
        # Check token length and truncate if necessary
        total_tokens = len(self.encoding.encode(prompt))
        if total_tokens > self.max_context_length:
            # Truncate the prompt to fit context length
            tokens = self.encoding.encode(prompt)
            truncated_tokens = tokens[:self.max_context_length]
            prompt = self.encoding.decode(truncated_tokens)

        return {
            "custom_id": str(item_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
            },
        }

    async def submit_batch(self, ids: List[str], prompts: List[str], batch_name: str) -> List[str]:
        """Submit items for batch processing"""
        batch_ids = []
        
        # Create batch requests
        requests = [self._create_batch_request(id, prompt) for id, prompt in zip(ids, prompts)]
        
        # Split requests into chunks
        for i, chunk_start in enumerate(range(0, len(requests), self.batch_chunk_size)):
            chunk = requests[chunk_start:chunk_start + self.batch_chunk_size]
            
            # Create JSONL content
            jsonl_content = "\n".join(json.dumps(request) for request in chunk)
            
            # Create temporary file
            temp_file = Path(f"{batch_name}_{i}.jsonl")
            temp_file.write_text(jsonl_content)
            
            try:
                # Upload file to OpenAI
                with open(temp_file, "rb") as f:
                    batch_input_file = self.client.files.create(
                        file=f,
                        purpose="batch"
                    )
                
                # Create batch
                batch = self.client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window=self.completion_window,
                    metadata={
                        "description": f"{batch_name} - Chunk {i}"
                    }
                )
                
                batch_ids.append(batch.id)
                
            finally:
                # Cleanup temporary file
                temp_file.unlink(missing_ok=True)
        
        return batch_ids

    async def get_batch_results(self, batch_id: str) -> Optional[List[CategoryResult]]:
        """Retrieve and parse results from a completed batch job"""
        try:
            # Get batch status
            batch = self.client.batches.retrieve(batch_id)
            
            if batch.status == "completed":
                # Get output file content
                output_file = self.client.files.content(batch.output_file_id)
                file_lines = output_file.text.strip().split('\n')
                
                # Parse JSON lines
                results = []
                for line in file_lines:
                    try:
                        response_data = json.loads(line)
                        item_id = response_data['custom_id']
                        model_response = response_data['response']['body']['choices'][0]['message']['content']

                        # Create CategoryResult object
                        result = CategoryResult(
                            id=item_id,
                            categories=[],  # Will be parsed from model_response
                            model_response={"raw_response": model_response},
                            processed_at=datetime.now()
                        )
                        results.append(result)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing response line: {e}")
                        continue
                
                return results
                
            elif batch.status in ["failed", "cancelled"]:
                logger.error(f"Batch {batch_id} {batch.status}")
                return None
                
            else:
                # Batch still processing
                logger.info(f"Batch {batch_id} still {batch.status}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving batch results: {e}")
            return None

    async def categorize(self, items: List[TextItem], categories: List[str]) -> List[CategoryResult]:
        """Synchronously categorize items using OpenAI API"""
        results = []
        for item in items:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": item.text}],
                    max_tokens=self.max_tokens
                )
                
                result = CategoryResult(
                    id=item.id,
                    categories=[],  # Will be parsed from response
                    model_response={"raw_response": response.choices[0].message.content},
                    processed_at=datetime.now()
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error categorizing item {item.id}: {e}")
                continue
                
        return results 