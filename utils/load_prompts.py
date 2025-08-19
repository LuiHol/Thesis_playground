import yaml
from pathlib import Path
from config import PROMPTS_DIR, SHARED_DIR


def load_prompt(module_name: str) -> dict:
    prompt_file = PROMPTS_DIR / f'{module_name}.yaml'

    if not prompt_file.exists():
        raise FileNotFoundError(f'Prompt file not found: {prompt_file}')

    with open(prompt_file, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing {prompt_file}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary in YAML file, got {type(data)}")

    return data

def load_yaml(filename: str) -> dict:
    """
    Load a YAML file from the shared directory.
    Example: load_yaml("event_mappings.yaml")
    """
    file_path = SHARED_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
