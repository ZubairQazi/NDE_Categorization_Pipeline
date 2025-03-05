# llm/openai_provider.py

from typing import List, Optional, Dict, Tuple
import asyncio
from datetime import datetime
import logging
import json
import os
from pathlib import Path
import sqlite3
import tiktoken
from openai import OpenAI
from ..core.data_model import TextItem, CategoryResult, BatchJob, JobStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)

class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 248,
        max_context_length: int = 128000,
        batch_chunk_size: int = 50000,
        completion_window: str = "24h",
        storage_dir: str = "storage"
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
        
        # Initialize storage
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._init_storage()

    def _init_storage(self):
        """Initialize SQLite database for storing batch information"""
        db_path = self.storage_dir / "batches.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    job_name TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata TEXT
                )
            """)

    def _store_batch_info(self, batch_id: str, job_name: str, metadata: dict):
        """Store batch information in SQLite database"""
        db_path = self.storage_dir / "batches.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO batches (batch_id, job_name, status, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (batch_id, job_name, "pending", datetime.now(), json.dumps(metadata))
            )

    def update_batch_status(self, batch_id: str, status: str):
        """Update batch status in database"""
        db_path = self.storage_dir / "batches.db"
        with sqlite3.connect(db_path) as conn:
            if status == "completed":
                conn.execute(
                    "UPDATE batches SET status = ?, completed_at = ? WHERE batch_id = ?",
                    (status, datetime.now(), batch_id)
                )
            else:
                conn.execute(
                    "UPDATE batches SET status = ? WHERE batch_id = ?",
                    (status, batch_id)
                )

    def get_batch_info(self, batch_id: str = None, job_name: str = None) -> List[Dict]:
        """Retrieve batch information from database"""
        db_path = self.storage_dir / "batches.db"
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            if batch_id:
                rows = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchall()
            elif job_name:
                rows = conn.execute("SELECT * FROM batches WHERE job_name = ?", (job_name,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM batches").fetchall()
            return [dict(row) for row in rows]

    def _split_jsonl_file(self, filepath: str, max_size_mb: int = 100) -> List[str]:
        """Split a JSONL file if it exceeds max_size_mb"""
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        if file_size_mb <= max_size_mb:
            return [filepath]
            
        # Read all lines from the file
        with open(filepath, "r") as f:
            lines = f.readlines()
            
        # Split into two parts
        mid = len(lines) // 2
        base_path = os.path.splitext(filepath)[0]
        
        # Create two new files
        new_files = []
        for idx, chunk in enumerate([lines[:mid], lines[mid:]]):
            new_filepath = f"{base_path}_{chr(97 + idx)}.jsonl"  # Adds 'a' or 'b' suffix
            with open(new_filepath, "w") as f:
                f.writelines(chunk)
            new_files.append(new_filepath)
            
        # Delete the original file
        os.remove(filepath)
        
        # Recursively check and split the new files if needed
        final_files = []
        for new_file in new_files:
            final_files.extend(self._split_jsonl_file(new_file, max_size_mb))
            
        return final_files

    async def check_existing_batches(self) -> bool:
        """Check for existing batch jobs and return whether to proceed"""
        existing_batches = self.client.batches.list()
        if existing_batches:
            running_batch_count = sum(1 for batch in existing_batches.data 
                                    if batch.status in ["in_progress", "finalizing", "validating", "cancelling"])
            if running_batch_count > 0:
                logger.warning(f"There are {running_batch_count} running batch jobs.")
                return False
        return True

    def _create_batch_request(self, item: TextItem, prompt_template: str) -> Dict:
        """Create a single batch request for an item"""
        # Prepare the prompt
        prompt = prompt_template.replace("<title>", item.metadata.get("title", "No Title"))
        prompt = prompt.replace("<abstract>", item.text)
        
        # Check token length and truncate if necessary
        total_tokens = len(self.encoding.encode(prompt))
        if total_tokens > self.max_context_length:
            tokens = self.encoding.encode(item.text)
            excess_tokens = total_tokens - self.max_context_length
            truncated_tokens = tokens[:-excess_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            prompt = prompt_template.replace("<title>", item.metadata.get("title", "No Title"))
            prompt = prompt.replace("<abstract>", truncated_text)

        return {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
            },
        }

    async def submit_batch(self, items: List[TextItem], prompt_template: str, job_name: str) -> List[str]:
        """Submit items for batch processing"""
        batch_ids = []
        temp_dir = self.storage_dir / "temp_batches"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Create batch requests
            requests = [self._create_batch_request(item, prompt_template) for item in items]
            
            # Split requests into chunks
            for i, chunk_start in enumerate(range(0, len(requests), self.batch_chunk_size)):
                chunk = requests[chunk_start:chunk_start + self.batch_chunk_size]
                
                # Create initial JSONL file
                initial_file = temp_dir / f"{job_name}_chunk_{i}.jsonl"
                with open(initial_file, "w") as f:
                    for request in chunk:
                        f.write(json.dumps(request) + "\n")
                
                # Split file if needed
                split_files = self._split_jsonl_file(str(initial_file))
                
                # Process each file (original or split)
                for file_path in split_files:
                    # Extract suffix for unique identification
                    suffix = Path(file_path).stem.split("_")[-1]
                    
                    # Upload file to OpenAI
                    with open(file_path, "rb") as f:
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
                            "description": f"{job_name} - {suffix}",
                            "original_file": file_path
                        }
                    )
                    
                    # Store batch information
                    self._store_batch_info(
                        batch_id=batch.id,
                        job_name=job_name,
                        metadata={
                            "file_path": file_path,
                            "chunk_index": i,
                            "suffix": suffix
                        }
                    )
                    
                    batch_ids.append(batch.id)
                    
        finally:
            # Cleanup temporary files
            for file in temp_dir.glob(f"{job_name}_*.jsonl"):
                file.unlink(missing_ok=True)
            
        return batch_ids

    async def get_batch_results(self, batch_id: str) -> Optional[List[CategoryResult]]:
        """Retrieve results for a batch job"""
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status == "completed":
            self.update_batch_status(batch_id, "completed")
            
            # Get output file
            output_file = self.client.files.retrieve(batch.output_file_id)
            
            # Download and process results
            results = []
            # Process the output file and create CategoryResult objects
            # TODO: Parsing logic
            
            return results
            
        elif batch.status in ["failed", "cancelled"]:
            self.update_batch_status(batch_id, batch.status)
            logger.error(f"Batch {batch_id} {batch.status}")
            return None
        else:
            # Batch still processing
            return None

    async def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of all batches for a specific job"""
        batches = self.get_batch_info(job_name=job_name)
        
        total_batches = len(batches)
        completed = sum(1 for b in batches if b["status"] == "completed")
        failed = sum(1 for b in batches if b["status"] == "failed")
        pending = total_batches - completed - failed
        
        return {
            "job_name": job_name,
            "total_batches": total_batches,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "batches": batches
        }