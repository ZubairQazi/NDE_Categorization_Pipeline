# Abstract base class for LLM providers
from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.data_model import TextItem, CategoryResult, BatchJob

class LLMProvider(ABC):
    @abstractmethod
    def categorize(self, items: List[TextItem], categories: List[str]) -> List[CategoryResult]:
        """Synchronously categorize items"""
        pass

    @abstractmethod
    async def batch_categorize(self, items: List[TextItem], categories: List[str], batch_name: str) -> List[CategoryResult]:
        """Submit items for batch processing and return results"""
        pass