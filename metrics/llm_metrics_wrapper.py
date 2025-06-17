from datetime import datetime
from typing import List, Optional
from pipeline.llm.base import LLMProvider
from pipeline.core.data_model import TextItem, CategoryResult, BatchJob

class LLMMetricsWrapper(LLMProvider):
    """Wrapper for LLM providers that adds metrics collection"""
    
    def __init__(self, provider: LLMProvider, metrics_collector):
        self.provider = provider
        self.metrics_collector = metrics_collector
    
    def categorize(self, items: List[TextItem], categories: List[str]) -> List[CategoryResult]:
        """Categorize items with metrics collection (sync)"""
        start_time = datetime.now()
        try:
            results = self.provider.categorize(items, categories)
            
            # Record metrics for each item
            for item, result in zip(items, results):
                completion = getattr(result, "model_response", {}).get("raw_response", str(result.categories))
                self.metrics_collector.record_request(
                    request_id=item.id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    prompt=item.text,
                    completion=completion
                )
            
            return results
            
        except Exception as e:
            # Record error metrics
            for item in items:
                self.metrics_collector.record_request(
                    request_id=item.id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    prompt=item.text,
                    completion="",
                    error=str(e)
                )
            raise
    
    async def batch_categorize(self, items: List[TextItem], categories: List[str], batch_name: str) -> List[CategoryResult]:
        """Batch categorize items with metrics collection (async)"""
        start_time = datetime.now()
        try:
            results = await self.provider.batch_categorize(items, categories, batch_name)
            
            # Record batch metrics
            completions = [getattr(result, "model_response", {}).get("raw_response", str(result.categories)) for result in results]
            self.metrics_collector.record_batch(
                batch_id=batch_name,
                num_samples=len(items),
                start_time=start_time,
                end_time=datetime.now(),
                prompts=[item.text for item in items],
                completions=completions
            )
            
            return results
            
        except Exception as e:
            # Record error metrics
            self.metrics_collector.record_batch(
                batch_id=batch_name,
                num_samples=len(items),
                start_time=start_time,
                end_time=datetime.now(),
                error=str(e)
            )
            raise 