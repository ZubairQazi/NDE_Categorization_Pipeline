from typing import Dict, List
from ..core.data_model import TextItem, CategoryResult
from ..processors.base import DataProcessor

class PromptFormatter(DataProcessor):
    def __init__(self, template: str, field_mappings: Dict[str, str]):
        """
        Initialize prompt formatter
        
        Args:
            template: Prompt template with placeholders like <title>
            field_mappings: Dict mapping template fields to TextItem attributes
                e.g. {"title": "metadata.title", "abstract": "text"}
        """
        self.template = template
        self.field_mappings = field_mappings

    def process_input(self, items: List[TextItem]) -> List[TextItem]:
        for item in items:
            prompt = self.template
            for template_field, item_field in self.field_mappings.items():
                # Handle nested metadata fields
                if item_field.startswith("metadata."):
                    value = item.metadata.get(item_field.split(".", 1)[1], "")
                else:
                    value = getattr(item, item_field, "")
                prompt = prompt.replace(f"<{template_field}>", str(value))
            item.text = prompt
        return items

    def process_output(self, results: List[CategoryResult]) -> List[CategoryResult]:
        return results 