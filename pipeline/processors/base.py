# Abstract base class for data processors
from abc import ABC, abstractmethod
from typing import List
from ..core.data_model import TextItem, CategoryResult

class DataProcessor(ABC):
    @abstractmethod
    def process_input(self, items: List[TextItem]) -> List[TextItem]:
        """Process input data before categorization"""
        pass
    
    @abstractmethod
    def process_output(self, results: List[CategoryResult]) -> List[CategoryResult]:
        """Process results after categorization"""
        pass