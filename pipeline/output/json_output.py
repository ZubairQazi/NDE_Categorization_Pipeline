# JSON output implementation
import json
from pathlib import Path
from typing import List

from ..core.data_model import CategoryResult
from .base import DataOutput


class JSONOutput(DataOutput):
    def __init__(self, output_dir: Path, filename: str):
        self.output_path = output_dir / filename
        self.output_dir = output_dir
        self._initialized = False

    def write(self, results: List[CategoryResult]) -> None:
        """Write results to JSON file as a single array"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to dictionaries
        results_data = [
            {
                "id": result.id,
                "categories": result.categories,
                "confidence_scores": result.confidence_scores,
                "model_response": result.model_response,
                "processed_at": (
                    result.processed_at.isoformat() if result.processed_at else None
                ),
            }
            for result in results
        ]

        if not self.output_path.exists():
            # Create new file with the first batch of results
            with open(self.output_path, "w") as f:
                json.dump(results_data, f, indent=2)
            self._initialized = True
        else:
            # Append to existing JSON array
            self._append_to_json_array(results_data)

    def _append_to_json_array(self, new_data: List[dict]) -> None:
        """Append new data to existing JSON array file"""
        try:
            # Read existing data
            with open(self.output_path, "r") as f:
                existing_data = json.load(f)

            # Ensure existing data is a list
            if not isinstance(existing_data, list):
                existing_data = [existing_data]

            # Append new data
            existing_data.extend(new_data)

            # Write back to file
            with open(self.output_path, "w") as f:
                json.dump(existing_data, f, indent=2)

        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or doesn't exist, create new
            with open(self.output_path, "w") as f:
                json.dump(new_data, f, indent=2)
