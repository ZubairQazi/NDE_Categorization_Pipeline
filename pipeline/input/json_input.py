# JSON input implementation
from .base import DataInput
from ..core import TextItem 

import pandas as pd
import json
from typing import Optional, Dict, Any, List


class JSONInput(DataInput):
    def __init__(
        self, 
        filepath: str,
        text_columns: Optional[List[str]] = None,
        metadata_mapping: Optional[Dict[str, str]] = None,
        text_separator: str = " "
    ):
        """
        Initialize JSON input processor
        
        Args:
            filepath: Path to JSON file
            text_columns: List of columns to combine into the text field
            metadata_mapping: Dict mapping JSON fields to metadata keys
                e.g. {"name": "title", "url": "source_url"}
            text_separator: String to use when joining text fields
        """
        self.filepath = filepath
        self.text_columns = text_columns or ["description"]
        self.metadata_mapping = metadata_mapping or {}
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

    def get_text_items(self) -> List[TextItem]:
        if "_id" not in self.dataset.columns:
            # Create sequential IDs if none provided
            self.dataset["_id"] = range(len(self.dataset))

        self.dataset["_id"] = self.dataset["_id"].fillna("").astype(str)

        return [
            TextItem(
                id=row["_id"],
                text=self._combine_text(row),
                metadata=self._extract_metadata(row)
            )
            for _, row in self.dataset.iterrows()
        ]
