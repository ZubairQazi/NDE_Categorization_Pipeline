# llm/openai_provider.py
from typing import List, Optional, Dict, Tuple, Any
import asyncio
from datetime import datetime
import logging
import json
import os
from pathlib import Path
import sqlite3
import tiktoken
from openai import OpenAI
import threading
from contextlib import contextmanager
import tempfile
import uuid
from filelock import FileLock
from ..core.data_model import TextItem, CategoryResult, BatchJob, JobStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)

class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
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
        
        # Initialize storage with process-safe setup
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / "batches.db"
        self.db_lock_path = self.storage_dir / "batches.db.lock"
        self._init_storage()
        
        # Create a thread-local storage for database connections
        self._local = threading.local()

    @contextmanager
    def _get_db_connection(self):
        """Process and thread-safe database connection context manager"""
        with FileLock(self.db_lock_path):
            if not hasattr(self._local, 'connection'):
                self._local.connection = sqlite3.connect(self.db_path)
                self._local.connection.row_factory = sqlite3.Row
            
            try:
                yield self._local.connection
            finally:
                self._local.connection.close()
                delattr(self._local, 'connection')

    def _init_storage(self):
        """Initialize SQLite database with process-safe access"""
        with FileLock(self.db_lock_path):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS batches (
                        batch_id TEXT PRIMARY KEY,
                        job_name TEXT,
                        process_id TEXT,
                        status TEXT,
                        created_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        metadata TEXT
                    )
                """)

    def _get_process_safe_temp_dir(self, job_name: str) -> Path:
        """Create a process-safe temporary directory"""
        process_id = str(os.getpid())
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = self.storage_dir / "temp_batches" / f"{job_name}_{process_id}_{unique_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _store_batch_info(self, batch_id: str, job_name: str, metadata: dict):
        """Store batch information with process ID"""
        process_id = str(os.getpid())
        with self._get_db_connection() as conn:
            conn.execute(
                """INSERT INTO batches 
                   (batch_id, job_name, process_id, status, created_at, metadata) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (batch_id, job_name, process_id, "pending", datetime.now(), json.dumps(metadata))
            )
            conn.commit()

    def update_batch_status(self, batch_id: str, status: str):
        """Update batch status with process-safe access"""
        with self._get_db_connection() as conn:
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

    def get_batch_info(self, batch_id: str = None, job_name: str = None, process_id: str = None) -> List[Dict]:
        """Retrieve batch information with optional process ID filter"""
        with self._get_db_connection() as conn:
            if batch_id:
                query = "SELECT * FROM batches WHERE batch_id = ?"
                params = (batch_id,)
            elif job_name and process_id:
                query = "SELECT * FROM batches WHERE job_name = ? AND process_id = ?"
                params = (job_name, process_id)
            elif job_name:
                query = "SELECT * FROM batches WHERE job_name = ?"
                params = (job_name,)
            else:
                query = "SELECT * FROM batches"
                params = ()
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def _split_jsonl_file(self, filepath: str, max_size_mb: int = 100) -> List[str]:
        """Split a JSONL file with process-safe file handling"""
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        if file_size_mb <= max_size_mb:
            return [filepath]
        
        # Create process-safe temporary directory for splits
        temp_dir = Path(tempfile.mkdtemp(dir=self.storage_dir / "splits"))
        
        try:
            # Read all lines from the file
            with open(filepath, "r") as f:
                lines = f.readlines()
            
            # Split into two parts
            mid = len(lines) // 2
            new_files = []
            
            # Create two new files in the temporary directory
            for idx, chunk in enumerate([lines[:mid], lines[mid:]]):
                new_filepath = temp_dir / f"{Path(filepath).stem}_{chr(97 + idx)}.jsonl"
                with open(new_filepath, "w") as f:
                    f.writelines(chunk)
                new_files.append(str(new_filepath))
            
            # Recursively split if needed
            final_files = []
            for new_file in new_files:
                final_files.extend(self._split_jsonl_file(new_file, max_size_mb))
            
            return final_files
            
        finally:
            # Clean up original file and temporary directory if empty
            os.remove(filepath)
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

    async def submit_batch(self, items: List[TextItem], prompt_template: str, job_name: str) -> List[str]:
        """Submit items for batch processing with process-safe handling"""
        batch_ids = []
        temp_dir = self._get_process_safe_temp_dir(job_name)
        
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
                
                # Process each file
                for file_path in split_files:
                    # Upload and create batch
                    with open(file_path, "rb") as f:
                        batch_input_file = self.client.files.create(
                            file=f,
                            purpose="batch"
                        )
                    
                    batch = self.client.batches.create(
                        input_file_id=batch_input_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window=self.completion_window,
                        metadata={
                            "description": f"{job_name} - Process {os.getpid()}",
                            "original_file": file_path
                        }
                    )
                    
                    self._store_batch_info(
                        batch_id=batch.id,
                        job_name=job_name,
                        metadata={
                            "file_path": file_path,
                            "chunk_index": i,
                            "process_id": str(os.getpid())
                        }
                    )
                    
                    batch_ids.append(batch.id)
                    
        finally:
            # Cleanup temporary files
            if temp_dir.exists():
                for file in temp_dir.glob("*.jsonl"):
                    file.unlink(missing_ok=True)
                temp_dir.rmdir()
            
        return batch_ids

    async def get_job_status(self, job_name: str, process_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of batches for a job, optionally filtered by process ID"""
        if process_id:
            batches = self.get_batch_info(job_name=job_name, process_id=process_id)
        else:
            batches = self.get_batch_info(job_name=job_name)
        
        total_batches = len(batches)
        completed = sum(1 for b in batches if b["status"] == "completed")
        failed = sum(1 for b in batches if b["status"] == "failed")
        pending = total_batches - completed - failed
        
        return {
            "job_name": job_name,
            "process_id": process_id,
            "total_batches": total_batches,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "batches": batches
        }