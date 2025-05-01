from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import asyncio
import os
import torch
from pathlib import Path
import re

from ..core.data_model import TextItem, CategoryResult, BatchJob, JobStatus
from .base import LLMProvider

logger = logging.getLogger(__name__)

class HFLocalModel:
    """Handles local HuggingFace model inference"""
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: str = None,
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        cache_dir: str = None
    ):
        """Initialize a local HuggingFace model.
        
        Args:
            model_name_or_path: Name or path of the model to load
            device: Device to run the model on (e.g., "cuda", "cpu", "cuda:0")
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            cache_dir: Directory to cache downloaded models
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "The transformers package is required to use the HuggingFace local provider. "
                "Please install it with `pip install transformers`"
            )
            
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Determine dtype
        if torch_dtype == "auto":
            if "cuda" in device and torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32
        
        # Configure quantization if needed
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            try:
                import bitsandbytes
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit
                )
            except ImportError:
                logger.warning("bitsandbytes not installed. Falling back to full precision.")
        
        # Load tokenizer and model
        logger.info(f"Loading model {model_name_or_path} on {device} with dtype {torch_dtype}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=device if "cuda" in device else None,
            cache_dir=cache_dir
        )
        
        if "cpu" in device:
            self.model = self.model.to(device)
        
        # Set generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info(f"Model loaded successfully")

    async def generate(self, prompt: str) -> str:
        """Generate text using the local model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated text response
        """
        try:
            # Run in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._generate_sync, prompt
            )
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def _generate_sync(self, prompt: str) -> str:
        """Synchronous version of generate for use with run_in_executor.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
            )
        
        # Decode the generated tokens
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):]
            
        return response.strip()

class HFLocalProvider(LLMProvider):
    """HuggingFace local model provider"""

    def __init__(
        self,
        model_name_or_path: str,
        prompt_template: str,
        device: str = None,
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_retries: int = 3,
        retry_delay: int = 5,
        cache_dir: str = None
    ):
        """Initialize the HuggingFace local provider.
        
        Args:
            model_name_or_path: Name or path of the model to load
            prompt_template: Template for formatting prompts
            device: Device to run the model on (e.g., "cuda", "cpu", "cuda:0")
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            cache_dir: Directory to cache downloaded models
        """
        self.model = HFLocalModel(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            cache_dir=cache_dir
        )
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Store batch jobs in memory
        self._batch_jobs: Dict[str, BatchJob] = {}

    async def categorize(self, items: List[TextItem], categories: List[str]) -> List[CategoryResult]:
        """Categorize items synchronously.
        
        Args:
            items: List of items to categorize
            categories: List of valid categories
            
        Returns:
            List of categorization results
        """
        results = []
        for item in items:
            # TODO: Check already formatted by the PromptFormatter preprocessor
            prompt = item.text
            
            for attempt in range(self.max_retries):
                try:
                    response = await self.model.generate(prompt)
                    
                    results.append(
                        CategoryResult(
                            id=item.id,
                            categories=[],
                            model_response={"raw_response": response},
                            processed_at=datetime.now()
                        )
                    )
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to categorize item {item.id} after {self.max_retries} attempts: {e}")
                        raise
                    await asyncio.sleep(self.retry_delay)
                    
        return results

    async def batch_categorize(self, ids: List[str], prompts: List[str], batch_name: str) -> List[str]:
        """Submit a batch of prompts for categorization.
        
        Args:
            ids: List of item IDs
            prompts: List of prompts to categorize
            batch_name: Name of the batch
            
        Returns:
            List of batch job IDs
        """
        # Create a batch job
        job_id = f"{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create TextItems from the IDs and prompts
        items = [TextItem(id=id, text=prompt) for id, prompt in zip(ids, prompts)]
        
        # Submit the batch
        job = await self.submit_batch(items, [])  # Categories will be extracted from prompts
        
        return [job.job_id]

    async def submit_batch(self, items: List[TextItem], categories: List[str]) -> BatchJob:
        """Submit items for batch processing.
        
        Args:
            items: List of items to process
            categories: List of valid categories
            
        Returns:
            BatchJob object representing the submitted job
        """
        # Create a new batch job
        job = BatchJob(
            job_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status=JobStatus.PENDING,
            items=items,
            created_at=datetime.now()
        )
        
        # Store the job
        self._batch_jobs[job.job_id] = job
        
        # Process the batch asynchronously
        asyncio.create_task(self._process_batch(job, categories))
        
        return job

    async def _process_batch(self, job: BatchJob, categories: List[str]) -> None:
        """Process a batch job asynchronously.
        
        Args:
            job: The batch job to process
            categories: List of valid categories
        """
        job.status = JobStatus.IN_PROGRESS
        
        try:
            # Process items in the batch
            results = await self.categorize(job.items, categories)
            
            # Update job with results
            job.results = results
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing batch {job.job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)

    async def get_batch_results(self, job_id: str) -> Optional[List[CategoryResult]]:
        """Get results for a batch job.
        
        Args:
            job_id: ID of the batch job
            
        Returns:
            List of categorization results if the job is complete, None otherwise
        """
        job = self._batch_jobs.get(job_id)
        if not job:
            logger.warning(f"Batch job {job_id} not found")
            return None
            
        if job.status == JobStatus.COMPLETED:
            return job.results
        elif job.status == JobStatus.FAILED:
            raise Exception(f"Batch job {job_id} failed: {job.error_message}")
        else:
            return None 