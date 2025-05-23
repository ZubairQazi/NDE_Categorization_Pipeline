# CSV input implementation
from .base import DataInput
from ..core import TextItem 

import pandas as pd
from typing import Optional, Dict, Any, List, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CSVInput(DataInput):
    def __init__(
        self, 
        filepath: str,
        text_columns: Optional[List[str]] = None,
        metadata_mapping: Optional[Dict[str, str]] = None,
        id_column: Optional[str] = None,
        text_separator: str = " "
    ):
        """
        Initialize CSV input processor
        
        Args:
            filepath: Path to CSV file
            text_columns: List of columns to combine into the text field
            metadata_mapping: Dict mapping CSV columns to metadata keys
                e.g. {"Name": "title", "URL": "source_url"}
            id_column: Name of the column to use as ID
            text_separator: String to use when joining text columns
        """
        self.filepath = filepath
        self.text_columns = text_columns or ["description"]
        self.metadata_mapping = metadata_mapping or {}
        self.id_column = id_column
        self.text_separator = text_separator
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        if not self.filepath.lower().endswith(".csv"):
            raise ValueError("Unsupported file format. Please provide a CSV file.")
        return pd.read_csv(self.filepath, lineterminator="\n")

    def _combine_text(self, row: pd.Series) -> str:
        """Helper method to combine multiple columns into one text field."""
        text_parts = [
            str(row[col]).strip() 
            for col in self.text_columns 
            if col in row and pd.notna(row[col])
        ]
        return self.text_separator.join(text_parts) if text_parts else "No Description"

    def _extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata from row using mapping"""
        metadata = {}
        for csv_col, meta_key in self.metadata_mapping.items():
            if csv_col in row and pd.notna(row[csv_col]):
                metadata[meta_key] = str(row[csv_col])
        return metadata

    def read(self) -> Iterator[TextItem]:
        """Read CSV file and yield TextItems"""
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
        """Check if CSV file exists and is valid"""
        try:
            if not Path(self.filepath).exists():
                return False
            # Try reading first row to validate format
            next(self.dataset.iterrows())
            return True
        except Exception as e:
            logger.error(f"Invalid CSV file: {e}")
            return False

if __name__ == "__main__":
    test_csv = "../tests/data/zenodo.csv"
    
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