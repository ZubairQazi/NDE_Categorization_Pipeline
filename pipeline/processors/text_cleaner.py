# Text cleaning processor
from typing import List, Optional
import re
from ..core.data_model import TextItem, CategoryResult
from ..processors.base import DataProcessor

class TextCleaner(DataProcessor):
    def __init__(
        self,
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        """
        Initialize text cleaner
        
        Args:
            remove_urls: Remove URLs from text
            remove_html: Remove HTML tags
            normalize_whitespace: Normalize all whitespace to single spaces
            min_length: Minimum text length (after cleaning)
            max_length: Maximum text length (after cleaning)
        """
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.min_length = min_length
        self.max_length = max_length
        
        # Common regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')

    def clean_text(self, text: str) -> str:
        """Clean a single text string"""
        if not text:
            return text

        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        if self.remove_html:
            text = self.html_pattern.sub(' ', text)
            
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
            
        if self.max_length:
            text = text[:self.max_length]
            
        if self.min_length and len(text) < self.min_length:
            text = ""
            
        return text

    def process_input(self, items: List[TextItem]) -> List[TextItem]:
        """Clean text before processing"""
        for item in items:
            item.text = self.clean_text(item.text)
        return [item for item in items if item.text]  # Remove empty items

    def process_output(self, results: List[CategoryResult]) -> List[CategoryResult]:
        """Pass through results unchanged"""
        return results