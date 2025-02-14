# CSV input implementation
from .base import DataInput
from ..core import TextItem 

import pandas as pd
from typing import Optional, Dict, Any, List


class CSVInput:
    def __init__(self, filepath: str, text_columns: Optional[List[str]] = None):
        self.filepath = filepath
        self.text_columns = text_columns
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        if not self.filepath.lower().endswith(".csv"):
            raise ValueError("Unsupported file format. Please provide a CSV file.")
        return pd.read_csv(self.filepath, lineterminator="\n")

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
            raise ValueError("CSV file must contain '_id' column.")

        self.dataset["_id"] = self.dataset["_id"].fillna("").astype(str)

        return [
            TextItem(id=row["_id"], text=self._combine_text(row))
            for _, row in self.dataset.iterrows()
        ]
    
if __name__ == "__main__":
    test_csv = "test_data.csv"
    
    # Creating a sample CSV file for testing
    sample_data = pd.DataFrame({
        "_id": ["1", "2", "3"],
        "name": ["Item A", "Item B", "Item C"],
        "description": ["Description A", "Description B", "Description C"]
    })
    sample_data.to_csv(test_csv, index=False)
    
    # Test with default behavior (using description only)
    csv_loader = CSVInput(test_csv)
    text_items = csv_loader.get_text_items()
    print("Default Behavior:")
    for item in text_items:
        print(item)
    
    # Test with combining name and description
    csv_loader_combined = CSVInput(test_csv, text_columns=["name", "description"])
    text_items_combined = csv_loader_combined.get_text_items()
    print("\nCombining Name and Description:")
    for item in text_items_combined:
        print(item)