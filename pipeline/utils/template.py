# utils/template.py
from pathlib import Path
from typing import List, Optional

class TemplateHandler:
    def __init__(self):
        # Store template directory relative to this file
        self.template_dir = Path(__file__).parent / "templates"
        self.topics_dir = Path(__file__).parent / "topics"

    def load_template(self, template_name: str) -> str:
        """Load template from file"""
        template_path = self.template_dir / template_name
        try:
            with open(template_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(f"Template file not found: {template_path}")

    def load_topics(self, topics_file: str) -> List[str]:
        """Load topics from file"""
        topics_path = self.topics_dir / topics_file
        try:
            with open(topics_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise ValueError(f"Topics file not found: {topics_path}")

    def format_template(self, template: str, topics: List[str], **kwargs) -> str:
        """Format template with topics and other parameters"""
        formatted_topics = "\n".join(topics)
        template = template.replace("<topics>", formatted_topics)
        
        # Replace any other template variables
        for key, value in kwargs.items():
            template = template.replace(f"<{key}>", str(value))
            
        return template
    
if __name__ == "__main__":
    template_handler = TemplateHandler()
    template = template_handler.load_template("prompt_template.txt")
    topics = template_handler.load_topics("edam_topics.txt")
    formatted_template = template_handler.format_template(template, topics)

    print(formatted_template)