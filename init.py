import os
import pathlib
from typing import Dict, Any

# Project structure with file contents
PROJECT_STRUCTURE = {
    "pipeline": {
        "__init__.py": "",
        "core": {
            "__init__.py": "",
            "pipeline.py": "# Pipeline orchestration\nfrom typing import List, Optional\n\nclass Pipeline:\n    pass",
            "data_model.py": "# Data structures and interfaces\nfrom dataclasses import dataclass\nfrom typing import List, Dict, Any, Optional\n\n@dataclass\nclass TextItem:\n    pass",
            "exceptions.py": "# Custom exceptions\n\nclass TextCategorizerError(Exception):\n    pass"
        },
        "input": {
            "__init__.py": "",
            "base.py": "# Abstract base class for input sources\nfrom abc import ABC, abstractmethod\n\nclass DataInput(ABC):\n    pass",
            "csv_input.py": "# CSV input implementation\nfrom .base import DataInput\n\nclass CSVInput(DataInput):\n    pass",
            "json_input.py": "# JSON input implementation\nfrom .base import DataInput\n\nclass JSONInput(DataInput):\n    pass"
        },
        "llm": {
            "__init__.py": "",
            "base.py": "# Abstract base class for LLM providers\nfrom abc import ABC, abstractmethod\n\nclass LLMProvider(ABC):\n    pass",
            "openai_provider.py": "# OpenAI implementation\nfrom .base import LLMProvider\n\nclass OpenAIProvider(LLMProvider):\n    pass",
            "hf_provider.py": "# HuggingFace implementation\nfrom .base import LLMProvider\n\nclass HuggingFaceProvider(LLMProvider):\n    pass"
        },
        "processors": {
            "__init__.py": "",
            "base.py": "# Abstract base class for data processors\nfrom abc import ABC, abstractmethod\n\nclass DataProcessor(ABC):\n    pass",
            "text_cleaner.py": "# Text cleaning processor\nfrom .base import DataProcessor\n\nclass TextCleaner(DataProcessor):\n    pass",
            "normalizer.py": "# Data normalization processor\nfrom .base import DataProcessor\n\nclass Normalizer(DataProcessor):\n    pass"
        },
        "output": {
            "__init__.py": "",
            "base.py": "# Abstract base class for output handlers\nfrom abc import ABC, abstractmethod\n\nclass DataOutput(ABC):\n    pass",
            "csv_output.py": "# CSV output implementation\nfrom .base import DataOutput\n\nclass CSVOutput(DataOutput):\n    pass",
            "json_output.py": "# JSON output implementation\nfrom .base import DataOutput\n\nclass JSONOutput(DataOutput):\n    pass"
        },
        "utils": {
            "__init__.py": "",
            "logging.py": "# Logging utilities\nimport logging\n\ndef get_logger(name):\n    return logging.getLogger(name)",
            "config.py": "# Configuration management\nfrom typing import Dict, Any\n\nclass Config:\n    pass"
        }
    }
}

def create_directory_structure(base_path: pathlib.Path, structure: Dict[str, Any]) -> None:
    """Recursively create directory structure and files."""
    for name, content in structure.items():
        path = base_path / name
        
        if isinstance(content, dict):
            # If content is a dict, it's a directory
            path.mkdir(parents=True, exist_ok=True)
            create_directory_structure(path, content)
        else:
            # If content is a string, it's a file
            path.write_text(content)

def main():
    # Get the current directory
    current_dir = pathlib.Path.cwd()
    
    # Create the project structure
    create_directory_structure(current_dir, PROJECT_STRUCTURE)
    
    print("Project structure created successfully!")
    print("\nCreated files and directories:")
    
    # Print the created structure
    for path in sorted(pathlib.Path('pipeline').rglob('*')):
        prefix = '    ' * (len(path.parts) - 1)
        if path.is_file():
            print(f"{prefix}📄 {path.name}")
        else:
            print(f"{prefix}📁 {path.name}")

if __name__ == "__main__":
    main()