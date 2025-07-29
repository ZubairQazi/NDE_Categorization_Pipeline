from datetime import datetime
from typing import List
import logging
import sys
import os

# Add the parent directory to the path so we can import from pipeline
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipeline.core.data_model import TextItem, CategoryResult
from pipeline.llm.base import LLMProvider
from .hf_collector import HuggingFaceMetricsCollector

logger = logging.getLogger(__name__)

class HFMetricsWrapper(LLMProvider):
    """Wrapper that collects metrics for HuggingFace providers"""
    
    def __init__(self, provider: LLMProvider, metrics_collector: HuggingFaceMetricsCollector):
        self.provider = provider
        self.metrics_collector = metrics_collector
        self.batch_counter = 0
    
    def categorize(self, items: List[TextItem], categories: List[str]) -> List[CategoryResult]:
        """Categorize items and collect metrics"""
        self.batch_counter += 1
        batch_id = f"batch_{self.batch_counter}"
        num_samples = len(items)
        
        logger.info(f"Starting batch {batch_id} with {num_samples} items")
        start_time = datetime.now()
        
        try:
            # Call the actual provider
            results = self.provider.categorize(items, categories)
            
            end_time = datetime.now()
            
            # Record successful batch
            self.metrics_collector.record_batch(
                batch_id=batch_id,
                num_samples=num_samples,
                start_time=start_time,
                end_time=end_time,
                cost=0.0  # No cost for local models
            )
            
            logger.info(f"Completed batch {batch_id} in {(end_time - start_time).total_seconds():.2f} seconds")
            return results
            
        except Exception as e:
            end_time = datetime.now()
            
            # Record failed batch
            self.metrics_collector.record_batch(
                batch_id=batch_id,
                num_samples=num_samples,
                start_time=start_time,
                end_time=end_time,
                error=str(e)
            )
            
            logger.error(f"Batch {batch_id} failed: {e}")
            raise
    
    async def batch_categorize(self, items: List[TextItem], categories: List[str], batch_name: str) -> List[CategoryResult]:
        """Forward to the wrapped provider"""
        return await self.provider.batch_categorize(items, categories, batch_name)
