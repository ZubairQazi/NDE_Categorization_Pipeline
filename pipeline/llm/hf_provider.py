# HuggingFace implementation
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import json
import asyncio

from ..core.data_model import TextItem, CategoryResult, BatchJob, JobStatus
from .base import LLMProvider

logger = logging.getLogger(__name__)

class HFInferenceAPI:
    """Handles communication with HuggingFace's inference API"""
    
    def __init__(self, api_token: str, model_name: str):
        """Initialize the HuggingFace inference API client.
        
        Args:
            api_token: HuggingFace API token
            model_name: Name of the model to use (e.g. "mistralai/Mistral-7B-Instruct-v0.2")
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "The huggingface_hub package is required to use the HuggingFace provider. "
                "Please install it with `pip install huggingface_hub`"
            )
            
        self.client = InferenceClient(token=api_token)
        self.model_name = model_name

    async def generate(self, prompt: str) -> str:
        """Generate text using the HuggingFace inference API.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The generated text response
        """
        try:
            response = await self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

class HFProvider(LLMProvider):
    """HuggingFace language model provider"""

    def __init__(
        self,
        api_token: str,
        model_name: str,
        prompt_template: str,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """Initialize the HuggingFace provider.
        
        Args:
            api_token: HuggingFace API token
            model_name: Name of the model to use
            prompt_template: Template for formatting prompts
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api = HFInferenceAPI(api_token, model_name)
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Store batch jobs in memory
        self._batch_jobs: Dict[str, BatchJob] = {}

    def _format_prompt(self, text: str, categories: List[str]) -> str:
        """Format the prompt using the template.
        
        Args:
            text: The text to categorize
            categories: List of valid categories
            
        Returns:
            The formatted prompt
        """
        categories_str = "\n".join(f"- {cat}" for cat in categories)
        return self.prompt_template.format(
            text=text,
            categories=categories_str
        )

    def _parse_response(self, response: str) -> List[str]:
        """Parse the model response into a list of categories.
        
        Args:
            response: Raw response from the model
            
        Returns:
            List of predicted categories
        """
        # Implement parsing logic based on your model's output format
        # This is just an example - adjust based on actual response format
        try:
            # Assuming response is a newline-separated list of categories
            categories = [cat.strip() for cat in response.split('\n') if cat.strip()]
            return categories
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []

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
            prompt = self._format_prompt(item.text, categories)
            
            for attempt in range(self.max_retries):
                try:
                    response = await self.api.generate(prompt)
                    predicted_categories = self._parse_response(response)
                    
                    results.append(CategoryResult(
                        id=item.id,
                        categories=predicted_categories,
                        model_response={"raw_response": response},
                        processed_at=datetime.now()
                    ))
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to categorize item {item.id} after {self.max_retries} attempts: {e}")
                        raise
                    await asyncio.sleep(self.retry_delay)
                    
        return results

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