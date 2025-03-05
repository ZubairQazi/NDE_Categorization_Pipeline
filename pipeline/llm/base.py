# Abstract base class for LLM providers
from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.data_model import TextItem, CategoryResult, BatchJob

class LLMProvider(ABC):
    @abstractmethod
    async def categorize(self, items: List[TextItem], categories: List[str]) -> List[CategoryResult]:
        """Synchronously categorize items"""
        pass

    @abstractmethod
    async def check_existing_batches(self) -> bool:
        """Check for existing batch jobs"""
        pass

    @abstractmethod
    async def batch_categorize(self, ids: List[str], prompts: List[str], batch_name: str) -> List[str]:
        """Submit items for batch processing"""
        pass

    @abstractmethod
    async def get_batch_results(self, batch_id: str) -> Optional[List[CategoryResult]]:
        """Retrieve results for a batch job"""
        pass