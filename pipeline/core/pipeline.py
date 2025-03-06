# core/pipeline.py
from typing import List, Optional, Type, Literal
import asyncio
from datetime import datetime

from .data_model import TextItem, CategoryResult, BatchJob, JobStatus
from ..input.base import DataInput
from ..llm.base import LLMProvider
from ..processors.base import DataProcessor
from ..output.base import DataOutput
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Pipeline:
    def __init__(
        self,
        input_handler: DataInput,
        llm_provider: LLMProvider,
        output_handler: DataOutput,
        preprocessors: Optional[List[DataProcessor]] = None,
        postprocessors: Optional[List[DataProcessor]] = None,
        categories: List[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: int = 5,
        mode: Literal["sync", "batch"] = "sync",
        existing_batch_ids: Optional[List[str]] = None
    ):
        self.input_handler = input_handler
        self.llm_provider = llm_provider
        self.output_handler = output_handler
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.categories = categories
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mode = mode
        self.active_jobs: List[BatchJob] = []

    def _process_input(self, items: List[TextItem]) -> List[TextItem]:
        """Apply all input processors to the data"""
        processed_items = items
        for processor in self.preprocessors:
            processed_items = processor.process_input(processed_items)
        return processed_items

    def _process_output(self, results: List[CategoryResult]) -> List[CategoryResult]:
        """Apply all output processors to the results"""
        processed_results = results
        for processor in self.postprocessors:
            processed_results = processor.process_output(processed_results)
        return processed_results

    async def process_items(self, items: List[TextItem]) -> Optional[List[CategoryResult]]:
        """Process items based on selected mode"""
        processed_items = self._process_input(items)
        
        # Print items that were filtered out during processing
        # TODO: Add warning log
        filtered_ids = set(item.id for item in items) - set(item.id for item in processed_items)
        if filtered_ids:
            logger.info(f"Items filtered during processing: {filtered_ids}")
            for item in items:
                if item.id in filtered_ids:
                    logger.info(f"Filtered item {item.id}: {item.text}")

        for attempt in range(self.max_retries):
            try:
                if self.mode == "sync":
                    # Process synchronously
                    results = await self.llm_provider.categorize(processed_items, self.categories)
                else:
                    # Process as batch
                    ids = [item.id for item in processed_items]
                    prompts = [item.text for item in processed_items]
                    batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    batch_ids = await self.llm_provider.batch_categorize(ids, prompts, batch_name)
                    results = []
                    
                    # Wait for batch completion
                    for batch_id in batch_ids:
                        while True:
                            batch_results = await self.llm_provider.get_batch_results(batch_id)
                            if batch_results is not None:
                                results.extend(batch_results)
                                break
                            await asyncio.sleep(self.retry_delay)
                
                return self._process_output(results)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to process items after {self.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(self.retry_delay)

    async def run(self) -> None:
        """Run the pipeline"""
        # Topic categories are necessary for any post processing
        if not self.categories:
            raise ValueError("Topic categories must be specified")
        
        # Continue with normal pipeline processing if input handler is provided
        if self.input_handler:
            if not self.input_handler.validate():
                raise ValueError("Invalid input source")
            
            batch = []
            for item in self.input_handler.read():
                batch.append(item)
                
                if len(batch) >= self.batch_size:
                    results = await self.process_items(batch)
                    if results:
                        self.output_handler.write(results)
                    batch = []
            
            # Process remaining items
            if batch:
                results = await self.process_items(batch)
                if results:
                    self.output_handler.write(results)