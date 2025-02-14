# utils/template.py
from typing import List, Dict, Any
from pathlib import Path

class TemplateHandler:
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)

    def load_template(self, template_name: str) -> str:
        """Load a template from file"""
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        return template_path.read_text()

    def load_topics(self, topics_file: str) -> List[str]:
        """Load topics from file"""
        topics_path = Path(topics_file)
        if not topics_path.exists():
            raise FileNotFoundError(f"Topics file not found: {topics_path}")
        
        with open(topics_path, "r") as f:
            topics = [line.strip() for line in f.readlines()]
        return topics

    def format_template(self, template: str, topics: List[str], **kwargs) -> str:
        """Format template with topics and other parameters"""
        formatted_topics = "\n".join(topics)
        template = template.replace("<topics>", formatted_topics)
        
        # Replace any other template variables
        for key, value in kwargs.items():
            template = template.replace(f"<{key}>", str(value))
            
        return template