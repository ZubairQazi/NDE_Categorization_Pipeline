from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime
import logging
import json
import os
from pathlib import Path
import sqlite3
import tiktoken
from openai import OpenAI
import uuid
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
        storage_dir: str = "storage",
        job_name: Optional[str] = None
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
        
        # Unique database per job to avoid conflicts
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / f"batches_{job_name or uuid.uuid4().hex}.db"
        self._init_storage()

    def _init_storage(self):
        """Initialize SQLite database for storing batch information"""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO batches (batch_id, job_name, status, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (batch_id, job_name, "pending", datetime.now(), json.dumps(metadata))
            )
            conn.commit()

    def update_batch_status(self, batch_id: str, status: str):
        """Update batch status in database"""
        with sqlite3.connect(self.db_path) as conn:
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
            conn.commit()

    def get_batch_info(self) -> List[Dict]:
        """Retrieve batch information from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM batches").fetchall()
            return [dict(row) for row in rows]

    async def submit_batch(self, items: List[TextItem], prompt_template: str, job_name: str) -> List[str]:
        """Submit items for batch processing"""
        batch_ids = []
        
        # Create batch requests
        requests = [
            {
                "custom_id": str(item.id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt_template.replace("<abstract>", item.text)}],
                    "max_tokens": self.max_tokens,
                },
            }
            for item in items
        ]
        
        # Submit batch to OpenAI
        for request in requests:
            batch = self.client.batches.create(
                input_file_id=request["custom_id"],
                endpoint="/v1/chat/completions",
                completion_window=self.completion_window,
                metadata={"description": job_name}
            )
            self._store_batch_info(batch.id, job_name, request)
            batch_ids.append(batch.id)
        
        return batch_ids

    async def get_batch_results(self, batch_id: str) -> Optional[List[CategoryResult]]:
        """Retrieve results for a batch job"""
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status == "completed":
            self.update_batch_status(batch_id, "completed")
            return []  # TODO: Implement result retrieval logic
        elif batch.status in ["failed", "cancelled"]:
            self.update_batch_status(batch_id, batch.status)
            logger.error(f"Batch {batch_id} {batch.status}")
            return None
        else:
            return None
