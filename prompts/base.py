#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base utilities for prompt templates in the LangGraph 멀티 에이전트 시스템
"""

from typing import Any, Dict, List, Optional, Union
import json
from pathlib import Path


def load_template_from_file(file_path: str) -> str:
    """Load a prompt template from a file"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {file_path}")
    
    return path.read_text(encoding='utf-8')


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with the given keyword arguments"""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing_key = str(e).strip("'")
        raise ValueError(f"Missing required parameter: {missing_key}")
    except Exception as e:
        raise ValueError(f"Error formatting template: {str(e)}")


def format_json_for_prompt(data: Any) -> str:
    """Format a Python object as JSON string for inclusion in a prompt"""
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract a JSON object from a response string"""
    # Find JSON-like content between ```json and ``` or { and }
    import re
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find content between braces
        brace_match = re.search(r'(\{[\s\S]*\})', response)
        if brace_match:
            json_str = brace_match.group(1)
        else:
            raise ValueError("No JSON content found in response")
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {str(e)}")


class PromptTemplateManager:
    """Manager for loading and using prompt templates"""
    
    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self.templates: Dict[str, str] = {}
        
        # Load system message if it exists
        system_path = self.template_dir / "system_message.txt"
        if system_path.exists():
            self.system_message = system_path.read_text(encoding='utf-8')
        else:
            self.system_message = ""
    
    def load_template(self, name: str) -> str:
        """Load a template by name"""
        if name in self.templates:
            return self.templates[name]
        
        file_path = self.template_dir / f"{name}.txt"
        if not file_path.exists():
            raise FileNotFoundError(f"Template not found: {name}")
        
        template = file_path.read_text(encoding='utf-8')
        self.templates[name] = template
        return template
    
    def format_system_message(self, **kwargs) -> str:
        """Format the system message with the given keyword arguments"""
        return format_prompt(self.system_message, **kwargs)
    
    def format_template(self, name: str, **kwargs) -> str:
        """Load and format a template by name"""
        template = self.load_template(name)
        return format_prompt(template, **kwargs) 