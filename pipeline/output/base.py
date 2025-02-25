# Abstract base class for output handlers
from abc import ABC, abstractmethod
from typing import List
from ..core.data_model import CategoryResult

class DataOutput(ABC):
    @abstractmethod
    def write(self, results: List[CategoryResult]) -> None:
        """Write results to output destination"""
        pass