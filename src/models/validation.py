from typing import Dict
from pydantic import BaseModel
from pydantic import ValidationError
from typing import Any
import json


class Parameter(BaseModel):
    type: str

class Returns(BaseModel):
    type: str

class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Returns


class Prompt(BaseModel):
    prompt: str

def _load_json(path: str, label: str) ->list[dict[str,Any]] :
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise RuntimeError(f"Error occurred while reading {path}: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON format in {path}: {e}")

    if not isinstance(data, list):
        raise RuntimeError(f"{label} file must be a JSON array")

    return data
    


def function_definition_check(path: str) -> list[Function]: ## receive the default path if none is typed 
    data = _load_json(path, "Function definitions")
    
    try:
        return [Function.model_validate(item) for item in data]
    except ValidationError as e:
        raise RuntimeError(f"Function definition validation error:\n{e}")


def prompt_check(path: str) -> list[Prompt]:
    data = _load_json(path, "Prompts")

    try:
        return [Prompt.model_validate(item) for item in data]
    except ValidationError as e:
        raise RuntimeError(f"Prompt validation error:\n{e}")
