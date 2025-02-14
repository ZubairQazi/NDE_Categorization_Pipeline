# core/data_model.py

# Pipeline orchestration
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class JobStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TextItem:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CategoryResult:
    id: str
    categories: List[str]
    confidence_scores: Optional[Dict[str, float]] = None
    model_response: Optional[Dict[str, Any]] = None
    processed_at: datetime = None

@dataclass
class BatchJob:
    job_id: str
    status: JobStatus
    items: List[TextItem]
    results: Optional[List[CategoryResult]] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# input/base.py
from abc import ABC, abstractmethod
from typing import Iterator, List
from .data_model import TextItem

class DataInput(ABC):
    @abstractmethod
    def read(self) -> Iterator[TextItem]:
        """Read data from source and yield TextItems"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate input source"""
        pass

# llm/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from .data_model import TextItem, CategoryResult, BatchJob

class LLMProvider(ABC):
    @abstractmethod
    async def categorize(self, items: List[TextItem], categories: List[str]) -> List[CategoryResult]:
        """Synchronously categorize items"""
        pass
    
    @abstractmethod
    async def submit_batch(self, items: List[TextItem], categories: List[str]) -> BatchJob:
        """Submit items for batch processing"""
        pass
    
    @abstractmethod
    async def get_batch_results(self, job: BatchJob) -> Optional[List[CategoryResult]]:
        """Retrieve results for a batch job"""
        pass

# processors/base.py
from abc import ABC, abstractmethod
from typing import List
from .data_model import TextItem, CategoryResult

class DataProcessor(ABC):
    @abstractmethod
    def process_input(self, items: List[TextItem]) -> List[TextItem]:
        """Process input data before categorization"""
        pass
    
    @abstractmethod
    def process_output(self, results: List[CategoryResult]) -> List[CategoryResult]:
        """Process results after categorization"""
        pass

# output/base.py
from abc import ABC, abstractmethod
from typing import List
from .data_model import CategoryResult

class DataOutput(ABC):
    @abstractmethod
    def write(self, results: List[CategoryResult]) -> None:
        """Write results to output destination"""
        pass