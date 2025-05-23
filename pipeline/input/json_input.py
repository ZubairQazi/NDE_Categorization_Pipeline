# JSON input implementation
from .base import DataInput
from ..core import TextItem 

import pandas as pd
import json
from typing import Optional, Dict, Any, List, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class JSONInput(DataInput):
    def __init__(
        self, 
        filepath: str,
        text_columns: Optional[List[str]] = None,
        metadata_mapping: Optional[Dict[str, str]] = None,
        id_column: Optional[str] = None,
        text_separator: str = " "
    ):
        """
        Initialize JSON input processor
        
        Args:
            filepath: Path to JSON file
            text_columns: List of columns to combine into the text field
            metadata_mapping: Dict mapping JSON fields to metadata keys
                e.g. {"name": "title", "url": "source_url"}
            id_column: Name of the column to use as ID
            text_separator: String to use when joining text fields
        """
        self.filepath = filepath
        self.text_columns = text_columns or ["description"]
        self.metadata_mapping = metadata_mapping or {}
        self.id_column = id_column
        self.text_separator = text_separator
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        if not self.filepath.lower().endswith(".json"):
            raise ValueError("Unsupported file format. Please provide a JSON file.")
        with open(self.filepath, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _combine_text(self, row: pd.Series) -> str:
        """Helper method to combine multiple columns into one text field."""
        text_parts = [
            str(row[col]).strip() for col in self.text_columns if col in row and pd.notna(row[col])
        ]
        return self.text_separator.join(text_parts) if text_parts else "No Description"

    def _extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata from row using mapping"""
        metadata = {}
        for json_field, meta_key in self.metadata_mapping.items():
            if json_field in row and pd.notna(row[json_field]):
                metadata[meta_key] = str(row[json_field])
        return metadata

    def read(self) -> Iterator[TextItem]:
        """Read JSON file and yield TextItems"""
        if self.id_column and self.id_column in self.dataset.columns:
            # Use the configured ID column
            id_series = self.dataset[self.id_column]
        else:
            # Create sequential IDs if no ID column is configured or found
            id_series = pd.Series(range(len(self.dataset)))
        
        id_series = id_series.fillna("").astype(str)
        
        for idx, row in self.dataset.iterrows():
            yield TextItem(
                id=id_series[idx],
                text=self._combine_text(row),
                metadata=self._extract_metadata(row)
            )

    def validate(self) -> bool:
        """Check if JSON file exists and is valid"""
        try:
            if not Path(self.filepath).exists():
                return False
            # Try reading first row to validate format
            next(self.dataset.iterrows())
            return True
        except Exception as e:
            logger.error(f"Invalid JSON file: {e}")
            return False
