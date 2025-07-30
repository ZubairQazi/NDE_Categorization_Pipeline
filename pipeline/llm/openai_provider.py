# llm/openai_provider.py

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken
from openai import OpenAI

from ..core.data_model import BatchJob, CategoryResult, JobStatus, TextItem
from ..utils.logging import get_logger
from .base import LLMProvider

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        model: str = "gpt-4.1-nano",
        max_tokens: int = 248,
        max_context_length: int = 128000,
        batch_chunk_size: int = 50000,
        completion_window: str = "24h",
    ):
        self.client = OpenAI(api_key=api_key, organization=org_id, project=project_id)
        self.model = model
        self.max_tokens = max_tokens
        self.max_context_length = max_context_length
        self.batch_chunk_size = batch_chunk_size
        self.completion_window = completion_window

        # Handle tiktoken encoding safely
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to o200k_base encoding for unknown models (default for newer models)
            logger.warning(f"Unknown model '{model}', using o200k_base encoding")
            self.encoding = tiktoken.get_encoding("o200k_base")

    async def check_existing_batches(self) -> bool:
        """Check for existing batch jobs and provide detailed information."""
        try:
            existing_batches = self.client.batches.list()
            if existing_batches.data:
                running_batches = [
                    batch
                    for batch in existing_batches.data
                    if batch.status
                    in ["in_progress", "finalizing", "validating", "cancelling"]
                ]
                completed_batches = [
                    batch
                    for batch in existing_batches.data
                    if batch.status == "completed"
                ]
                failed_batches = [
                    batch
                    for batch in existing_batches.data
                    if batch.status in ["failed", "cancelled"]
                ]

                if running_batches:
                    logger.warning(f"Found {len(running_batches)} running batch jobs:")
                    for batch in running_batches:
                        logger.warning(f"  - {batch.id}: {batch.status}")

                if completed_batches:
                    logger.info(f"Found {len(completed_batches)} completed batch jobs")

                if failed_batches:
                    logger.warning(
                        f"Found {len(failed_batches)} failed/cancelled batch jobs"
                    )

            return True  # Always returns True, so it won't block
        except Exception as e:
            logger.error(f"Error checking existing batches: {e}")
            return True

    def _create_batch_request(self, item_id: str, prompt: str) -> Dict:
        """Create a single batch request for an item"""
        # Check token length and truncate if necessary
        total_tokens = len(self.encoding.encode(prompt))
        if total_tokens > self.max_context_length:
            # Truncate the prompt to fit context length
            tokens = self.encoding.encode(prompt)
            truncated_tokens = tokens[: self.max_context_length]
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

    def _calculate_polling_interval(self, batch_age_minutes: int) -> int:
        """Calculate intelligent polling interval based on batch age"""
        if batch_age_minutes < 5:
            return 30  # 30 seconds for very new batches
        elif batch_age_minutes < 15:
            return 60  # 1 minute for recent batches
        elif batch_age_minutes < 60:
            return 300  # 5 minutes for batches under 1 hour
        else:
            return 600  # 10 minutes for older batches

    def _estimate_completion_time(
        self, batch_status: str, created_at: datetime
    ) -> Optional[str]:
        """Estimate completion time based on batch status and creation time"""
        now = datetime.now()
        elapsed = now - created_at

        if batch_status == "validating":
            return "2-5 minutes"
        elif batch_status == "in_progress":
            # Rough estimate: most batches complete within 24 hours
            if elapsed.total_seconds() < 3600:  # Less than 1 hour
                return "2-20 hours"
            elif elapsed.total_seconds() < 43200:  # Less than 12 hours
                return "2-12 hours"
            else:
                return "1-6 hours"
        elif batch_status == "finalizing":
            return "1-10 minutes"
        else:
            return None

    async def validate_batch_ids(
        self, batch_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Validate batch IDs and return valid and invalid ones"""
        valid_ids = []
        invalid_ids = []

        for batch_id in batch_ids:
            try:
                batch = self.client.batches.retrieve(batch_id)
                valid_ids.append(batch_id)
                logger.info(f"Batch {batch_id}: {batch.status}")
            except Exception as e:
                logger.error(f"Invalid batch ID {batch_id}: {e}")
                invalid_ids.append(batch_id)

        return valid_ids, invalid_ids

    async def get_batch_status_details(self, batch_id: str) -> Optional[Dict]:
        """Get detailed status information for a batch"""
        try:
            batch = self.client.batches.retrieve(batch_id)
            created_at = datetime.fromtimestamp(batch.created_at)

            details = {
                "id": batch.id,
                "status": batch.status,
                "created_at": created_at,
                "elapsed_time": datetime.now() - created_at,
                "request_counts": (
                    {
                        "total": batch.request_counts.total,
                        "completed": batch.request_counts.completed,
                        "failed": batch.request_counts.failed,
                    }
                    if batch.request_counts
                    else {}
                ),
                "metadata": batch.metadata or {},
            }

            if batch.status in ["in_progress", "finalizing", "validating"]:
                details["estimated_completion"] = self._estimate_completion_time(
                    batch.status, created_at
                )

            return details
        except Exception as e:
            logger.error(f"Error getting batch status for {batch_id}: {e}")
            return None

    async def wait_for_batch_completion(
        self, batch_ids: List[str], progress_callback=None
    ) -> Dict[str, List[CategoryResult]]:
        """Wait for batch completion with intelligent polling and progress reporting"""
        results = {}
        completed_batches = set()
        failed_batches = set()

        logger.info(f"Waiting for {len(batch_ids)} batches to complete...")

        while len(completed_batches) + len(failed_batches) < len(batch_ids):
            for batch_id in batch_ids:
                if batch_id in completed_batches or batch_id in failed_batches:
                    continue

                try:
                    batch_details = await self.get_batch_status_details(batch_id)
                    if not batch_details:
                        failed_batches.add(batch_id)
                        continue

                    status = batch_details["status"]

                    if progress_callback:
                        progress_callback(batch_id, batch_details)

                    if status == "completed":
                        logger.info(
                            f"Batch {batch_id} completed, retrieving results..."
                        )
                        batch_results = await self.get_batch_results(batch_id)
                        if batch_results:
                            results[batch_id] = batch_results
                            completed_batches.add(batch_id)
                        else:
                            logger.error(
                                f"Failed to retrieve results for completed batch {batch_id}"
                            )
                            failed_batches.add(batch_id)

                    elif status in ["failed", "cancelled"]:
                        logger.error(f"Batch {batch_id} {status}")
                        failed_batches.add(batch_id)

                    else:
                        # Calculate next polling interval
                        elapsed_minutes = (
                            batch_details["elapsed_time"].total_seconds() / 60
                        )
                        poll_interval = self._calculate_polling_interval(
                            elapsed_minutes
                        )

                        if batch_details.get("estimated_completion"):
                            logger.info(
                                f"Batch {batch_id}: {status}, ETA: {batch_details['estimated_completion']}"
                            )

                except Exception as e:
                    logger.error(f"Error checking batch {batch_id}: {e}")
                    await asyncio.sleep(30)  # Short delay before retrying

            # Wait before next polling cycle
            if len(completed_batches) + len(failed_batches) < len(batch_ids):
                # Use minimum polling interval from all active batches
                min_interval = 300  # Default 5 minutes
                await asyncio.sleep(min_interval)

        logger.info(
            f"Batch processing complete: {len(completed_batches)} completed, {len(failed_batches)} failed"
        )
        return results

    async def get_batch_results(self, batch_id: str) -> Optional[List[CategoryResult]]:
        """Retrieve results from a completed batch with enhanced error handling"""
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.status != "completed":
                logger.warning(
                    f"Batch {batch_id} is not completed (status: {batch.status})"
                )
                return None

            if not batch.output_file_id:
                logger.error(f"Batch {batch_id} has no output file")
                return None

            # Get output file content
            output_file = self.client.files.content(batch.output_file_id)
            file_lines = output_file.text.strip().split("\n")

            results = []
            errors = 0

            # Parse JSON lines
            for line_num, line in enumerate(file_lines, 1):
                try:
                    response_data = json.loads(line)

                    # Check if response has an error
                    if response_data.get("error"):
                        logger.error(
                            f"Batch {batch_id}, line {line_num}: API error - {response_data['error']}"
                        )
                        errors += 1
                        continue

                    item_id = response_data["custom_id"]
                    response_body = response_data["response"]["body"]

                    # Check for API errors in response body
                    if "error" in response_body:
                        logger.error(
                            f"Batch {batch_id}, item {item_id}: {response_body['error']}"
                        )
                        errors += 1
                        continue

                    model_response = response_body["choices"][0]["message"]["content"]
                    model_name = response_body.get("model", self.model)

                    # Extract logprobs if available
                    logprobs = response_body["choices"][0].get("logprobs", None)

                    # Create CategoryResult object
                    result = CategoryResult(
                        id=item_id,
                        categories=[],  # Will be parsed from model_response
                        model_response={
                            "raw_response": model_response,
                            "model_name": model_name,
                            "batch_id": batch_id,
                        },
                        confidence_scores=logprobs,
                        processed_at=datetime.now(),
                    )
                    results.append(result)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(
                        f"Batch {batch_id}, line {line_num}: Error parsing response - {e}"
                    )
                    errors += 1
                    continue

            logger.info(
                f"Retrieved {len(results)} results from batch {batch_id} ({errors} errors)"
            )
            return results

        except Exception as e:
            logger.error(f"Error retrieving batch results for {batch_id}: {e}")
            return None

    async def batch_categorize(
        self, items: List[TextItem], categories: List[str], batch_name: str
    ) -> List[CategoryResult]:
        """Submit items for batch processing and wait for results with enhanced monitoring"""
        # Check for existing batches
        await self.check_existing_batches()

        batch_ids = []
        all_results = []

        # Create batch requests
        requests = [self._create_batch_request(item.id, item.text) for item in items]
        logger.info(f"Created {len(requests)} batch requests for {len(items)} items")

        # Split requests into chunks respecting both count and size limits
        chunks = self._split_requests_by_size_and_count(requests, batch_name)
        logger.info(
            f"Split into {len(chunks)} chunks (respecting 50K count and 200MB size limits)"
        )

        chunk_counter = 0
        for i, chunk in enumerate(chunks):
            # Process chunk with adaptive splitting for oversized files
            sub_chunks = self._process_chunk_with_size_validation(
                chunk, batch_name, chunk_counter
            )

            for sub_chunk in sub_chunks:
                # Create JSONL content
                jsonl_content = "\n".join(json.dumps(request) for request in sub_chunk)

                # Create temporary file
                temp_file = Path(f"{batch_name}_{chunk_counter}.jsonl")
                temp_file.write_text(jsonl_content)

                # Final size check (should always pass now)
                file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
                logger.info(
                    f"Chunk {chunk_counter+1}: {len(sub_chunk)} requests, {file_size_mb:.1f} MB"
                )

                try:
                    # Upload file to OpenAI
                    with open(temp_file, "rb") as f:
                        batch_input_file = self.client.files.create(
                            file=f, purpose="batch"
                        )

                    # Create batch
                    batch = self.client.batches.create(
                        input_file_id=batch_input_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window=self.completion_window,
                        metadata={
                            "description": f"{batch_name} - Chunk {chunk_counter}",
                            "chunk_size": str(len(sub_chunk)),
                            "total_chunks": str(len(chunks)),
                        },
                    )

                    batch_ids.append(batch.id)
                    logger.info(
                        f"Created batch {batch.id} with {len(sub_chunk)} requests (chunk {chunk_counter+1})"
                    )

                except Exception as e:
                    logger.error(f"Error creating batch chunk {chunk_counter}: {e}")
                    raise
                finally:
                    # Cleanup temporary file
                    temp_file.unlink(missing_ok=True)

                chunk_counter += 1

        # Wait for all batches to complete with progress reporting
        def progress_callback(batch_id: str, details: Dict):
            elapsed = details["elapsed_time"]
            status = details["status"]
            eta = details.get("estimated_completion", "Unknown")
            logger.info(f"Batch {batch_id}: {status} (elapsed: {elapsed}, ETA: {eta})")

        logger.info(f"Submitted {len(batch_ids)} batches, waiting for completion...")
        batch_results = await self.wait_for_batch_completion(
            batch_ids, progress_callback
        )

        # Collect all results
        for batch_id, results in batch_results.items():
            all_results.extend(results)

        logger.info(f"Batch processing complete: {len(all_results)} total results")
        return all_results

    def categorize(
        self, items: List[TextItem], categories: List[str]
    ) -> List[CategoryResult]:
        """Synchronously categorize items using OpenAI API"""
        results = []
        logger.info(f"Processing {len(items)} items synchronously...")

        for i, item in enumerate(items, 1):
            try:
                # Show progress for larger batches, or at start/end for small ones
                if len(items) > 20:
                    if i % 10 == 0 or i == len(items):
                        logger.info(f"Processing item {i}/{len(items)}")
                elif len(items) > 5:
                    if i % 5 == 0 or i == len(items):
                        logger.info(f"Processing item {i}/{len(items)}")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": item.text}],
                    max_tokens=self.max_tokens,
                )

                result = CategoryResult(
                    id=item.id,
                    categories=[],  # Will be parsed from response
                    model_response={
                        "raw_response": response.choices[0].message.content,
                        "model_name": response.model,
                    },
                    confidence_scores=None,
                    processed_at=datetime.now(),
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error categorizing item {item.id}: {e}")
                continue

        logger.info(f"✓ Completed processing {len(results)}/{len(items)} items")
        return results

    def _split_requests_by_size_and_count(
        self, requests: List[dict], batch_name: str, max_size_mb: int = 200
    ) -> List[List[dict]]:
        """Split requests into chunks of 50K requests each"""
        chunks = []

        # Simple chunking by count - size validation happens later
        for i in range(0, len(requests), self.batch_chunk_size):
            chunk = requests[i : i + self.batch_chunk_size]
            chunks.append(chunk)

        logger.info(
            f"Split {len(requests)} requests into {len(chunks)} chunks of max {self.batch_chunk_size} requests each"
        )
        return chunks

    def _process_chunk_with_size_validation(
        self,
        chunk: List[dict],
        batch_name: str,
        chunk_index: int,
        max_size_mb: int = 200,
    ) -> List[List[dict]]:
        """Process a chunk and split it in half recursively until each sub-chunk is under 200MB"""
        max_size_bytes = max_size_mb * 1024 * 1024

        # Test the current chunk size
        jsonl_content = "\n".join(json.dumps(request) for request in chunk)
        current_size = len(jsonl_content.encode("utf-8"))

        if current_size <= max_size_bytes:
            # Chunk is fine, return as-is
            return [chunk]

        # Chunk is too large, split in half
        logger.warning(
            f"Chunk {chunk_index} is {current_size / (1024*1024):.1f} MB (>200MB). Splitting in half..."
        )

        mid_point = len(chunk) // 2
        if mid_point == 0:
            # Single request is too large - this shouldn't happen with normal text
            logger.error(
                f"Single request is larger than 200MB limit. Skipping request."
            )
            return []

        # Split in half and recursively process each half
        first_half = chunk[:mid_point]
        second_half = chunk[mid_point:]

        # Recursively process each half
        sub_chunks = []
        sub_chunks.extend(
            self._process_chunk_with_size_validation(
                first_half, batch_name, chunk_index, max_size_mb
            )
        )
        sub_chunks.extend(
            self._process_chunk_with_size_validation(
                second_half, batch_name, chunk_index, max_size_mb
            )
        )

        return sub_chunks
