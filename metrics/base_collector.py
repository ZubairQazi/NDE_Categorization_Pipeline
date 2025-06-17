from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

class BaseMetricsCollector(ABC):
    def __init__(self, output_dir: str = "metrics/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "total_samples": 0,
            "total_runtime": 0,
            "average_time_per_sample": 0,
            "total_cost": 0,
            "cost_per_sample": 0,
            "errors": [],
            "batch_metrics": [],
            "resource_usage": []
        }
    
    @abstractmethod
    def start_collection(self):
        """Start collecting metrics"""
        pass
    
    @abstractmethod
    def stop_collection(self):
        """Stop collecting metrics"""
        pass
    
    @abstractmethod
    def record_batch(self, batch_id: str, num_samples: int, start_time: datetime, end_time: datetime, 
                    cost: Optional[float] = None, error: Optional[str] = None):
        """Record metrics for a single batch"""
        pass
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save collected metrics to a JSON file"""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        return output_path 