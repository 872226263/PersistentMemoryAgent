# src/meta_service.py
import sys
import os
from typing import List, Dict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_services import model_manager

# This is the specific phrase we've identified from Chatbox logs.
# We can add more prefixes here if we identify other client behaviors.
META_PROMPT_PREFIXES = [
    "Based on the chat history, give this conversation a name."
]

def is_meta_prompt(text: str) -> bool:
    """
    Checks if a given text is a known meta-prompt from a client.
    """
    for prefix in META_PROMPT_PREFIXES:
        if text.strip().startswith(prefix):
            return True
    return False

def handle_meta_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Handles a meta-prompt by directly returning a hardcoded title.
    This avoids calling the LLM and prevents rate limit errors.
    """
    print("[MetaService] Handling meta-prompt with a hardcoded response.")
    # No longer calling the model, just return the placeholder title.
    return "PersistentMemoryAgent"
    
# This file doesn't need a singleton instance as it only contains utility functions. 