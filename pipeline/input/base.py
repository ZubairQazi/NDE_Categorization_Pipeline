# Abstract base class for input sources
from abc import ABC, abstractmethod
from typing import Iterator, List
from ..core.data_model import TextItem

class DataInput(ABC):
    @abstractmethod
    def read(self) -> Iterator[TextItem]:
        """Read data from source and yield TextItems"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate input source exists and is readable"""
        pass