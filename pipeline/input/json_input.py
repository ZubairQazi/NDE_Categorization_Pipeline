# JSON input implementation
from .base import DataInput
from ..core import TextItem 

import pandas as pd
import json
from typing import Optional, Dict, Any, List


class JSONInput(DataInput):
    def __init__(self, filepath: str, text_columns: Optional[List[str]] = None):
        self.filepath = filepath
        self.text_columns = text_columns
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        if not self.filepath.lower().endswith(".json"):
            raise ValueError("Unsupported file format. Please provide a JSON file.")
        with open(self.filepath, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _combine_text(self, row: pd.Series) -> str:
        """Helper method to combine multiple columns into one text field."""
        if self.text_columns:
            text_parts = [
                str(row[col]).strip() for col in self.text_columns if col in row and pd.notna(row[col])
            ]
            return " ".join(text_parts) if text_parts else "No Description"
        elif "description" in row and pd.notna(row["description"]):
            return str(row["description"]).strip()
        else:
            return "No Description"

    def get_text_items(self) -> List[TextItem]:
        if "_id" not in self.dataset.columns:
            raise ValueError("JSON file must contain '_id' column.")

        self.dataset["_id"] = self.dataset["_id"].fillna("").astype(str)

        return [
            TextItem(id=row["_id"], text=self._combine_text(row))
            for _, row in self.dataset.iterrows()
        ]