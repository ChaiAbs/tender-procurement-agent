"""Prompt loader — reads YAML files by agent name and returns the parsed dict."""

import os
import yaml
from config import PROMPTS_DIR


def load_prompt(agent_name: str) -> dict:
    """
    Load a prompt YAML file for the given agent.

    Args:
        agent_name: Filename without extension, e.g. "analysis_agent"

    Returns:
        Dict with at least "system" and "human" keys (plain strings).
        Edit the YAML file to change agent behaviour without touching Python code.
    """
    path = os.path.join(PROMPTS_DIR, f"{agent_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Expected a YAML file with 'system' and 'human' keys."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)
