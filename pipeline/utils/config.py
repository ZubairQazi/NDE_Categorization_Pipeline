# utils/config.py

from typing import Dict, Any, Optional
import json
from pathlib import Path

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.config_data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {self.config_path}")

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        return self.config_data["api_keys"]["openai"]

    @property
    def openai_org_id(self) -> Optional[str]:
        """Get OpenAI organization ID"""
        return self.config_data["project_org_ids"].get("openai_org")

    @property
    def openai_project_id(self) -> Optional[str]:
        """Get OpenAI project ID"""
        return self.config_data["project_org_ids"].get("openai_project")

    def get(self, key: str, default: Any = None) -> Any:
        """Get any configuration value by key"""
        keys = key.split(".")
        value = self.config_data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value