# JSON output implementation
from .base import DataOutput
from typing import List
import json
from pathlib import Path
from ..core.data_model import CategoryResult

class JSONOutput(DataOutput):
    def __init__(self, output_dir: Path, filename: str):
        self.output_path = output_dir / filename
        self.output_dir = output_dir
        
    def write(self, results: List[CategoryResult]) -> None:
        """Write results to JSON file"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dictionaries
        results_data = [
            {
                "id": result.id,
                "categories": result.categories,
                "confidence_scores": result.confidence_scores,
                "model_response": result.model_response,
                "processed_at": result.processed_at.isoformat() if result.processed_at else None
            }
            for result in results
        ]
        
        # Write or append to file
        mode = "w" if not self.output_path.exists() else "a"
        with open(self.output_path, mode) as f:
            json.dump(results_data, f, indent=2)