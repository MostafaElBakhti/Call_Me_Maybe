<div align="center">

*This project has been created as part of the 42 curriculum by mel-bakh.*

# Call Me Maybe

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Score](https://img.shields.io/badge/Score-125%2F100-2ecc71?style=flat-square)
![42](https://img.shields.io/badge/42-Project-black?style=flat-square)
![XP](https://img.shields.io/badge/XP-1764-orange?style=flat-square)
![Hours](https://img.shields.io/badge/Hours-80-blue?style=flat-square)
![Model](https://img.shields.io/badge/Model-Qwen3--0.6B-purple?style=flat-square)

**Translating natural language into structured JSON using token-level constrained decoding**

</div>

---

## Description

A function calling tool that translates natural language prompts into structured JSON function calls using a small LLM (Qwen3-0.6B) with constrained decoding.

Given a prompt like `"What is the sum of 40 and 2?"`, the system outputs:

```json
{
  "name": "fn_add_numbers",
  "parameters": { "a": 40, "b": 2 }
}
```

Instead of answering the question directly, it identifies the right function and extracts the correct arguments — bridging natural language and executable code.

---

## Instructions

**Requirements:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/)

**Install dependencies:**
```bash
uv sync
```

**Run with default paths:**
```bash
uv run python -m src
```

**Run with custom paths:**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --model Qwen/Qwen3-0.6B
```

**Lint:**
```bash
make lint
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/input/function_calling_tests.json` | Path to prompts file |
| `--functions_definition` | `data/input/functions_definition.json` | Path to function schemas |
| `--output` | `data/output/function_calling_results.json` | Path for results |
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace model name |

---

## Algorithm Explanation

The core technique is **constrained decoding** — guiding the model token by token instead of letting it generate freely.

```
"What is the sum of 2 and 3?"
        ↓  TOKENIZATION
[892] [318] [262] [4771] [286] [16] [17]
        ↓  LLM PROCESSING
logits → fn_add_numbers: 87% | fn_greet: 8% | null: 3%
        ↓  CONSTRAINED DECODING
only valid tokens allowed at each step
        ↓  OUTPUT
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

**Steps:**

1. Build a system prompt listing all available functions and their parameters.
2. Force the output to start with `{"name": "` — the model never sees a blank slate.
3. For the function name field, compute which tokens are valid at each step by comparing what has been generated so far against all known function name token sequences. Only tokens that continue a valid name are allowed.
4. Once the name is complete, switch to free generation but restricted to a filtered vocabulary containing only JSON-safe characters.
5. Stop when the output ends with `}}`.

This guarantees **100% valid JSON** regardless of model confidence.

---

## Design Decisions

| Decision | Reason |
|----------|--------|
| **Pydantic for validation** | Catches schema errors early with field-level messages before the model loads |
| **Prefix forcing** | Injecting `{"name": "` removes the biggest failure point — the model never decides how to start |
| **Vocabulary filtering** | Restricts free generation to JSON-relevant characters, prevents prose or unicode output |
| **Null function** | `"null"` is treated as a valid function name so the model can cleanly signal no match |
| **Two-phase generation** | Separate name phase and argument phase gives fine-grained control over each part |

---

## Performance Analysis

Tested on 30 prompts with Qwen3-0.6B on a Tesla T4 GPU:

| Metric | Result |
|--------|--------|
| Function selection accuracy | 90%+ |
| JSON validity | 100% |
| Processing time (30 prompts) | < 2 minutes on T4 GPU |

---

## Challenges Faced

- **Escape handling** — Regex parameters like `\d+` required special handling to survive JSON serialization without becoming `\\d+` in the output.
- **Tokenizer vocabulary** — Filtering the vocabulary to only JSON-safe tokens required understanding how Qwen3's tokenizer represents special characters.
- **Name token matching** — Function names can tokenize into multiple tokens, so the constrained decoding had to track partial matches across steps.

---

## Testing Strategy

- Tested with the provided sample prompts covering all function types.
- Tested null cases (weather, jokes, capital cities) to verify the model correctly returns `null` when no function matches.
- Tested edge cases: empty prompts, regex patterns, multi-token names.
- Validated all output JSON with `json.loads()` to confirm parseability.

---

## Bonus

**Multiple model support** — the `--model` flag accepts any HuggingFace model compatible with `llm_sdk`:

```bash
uv run python -m src --model Qwen/Qwen3-0.6B
uv run python -m src --model FrontiersMind/Nandi-Mini-150M-Tool-Calling
uv run python -m src --model FrontiersMind/Nandi-Mini-600M-Early-Checkpoint
```

**Generation visualization** — the token generation process is displayed live in the terminal:

```
[step 0]  name so far: fn_
[step 1]  name so far: fn_add_numbers
[step 5]  generating: {"name": "fn_add_numbers", "parameters": {"a": 2.0
[step 18] generating: {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

---

## Resources

- [Qwen3 Model — HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained Decoding — Outlines library concepts](https://github.com/outlines-dev/outlines)
- [Pydantic documentation](https://docs.pydantic.dev)
- [uv documentation](https://docs.astral.sh/uv/)

---

**AI usage:** Claude was used to help add type hints and docstrings to the existing code, fix flake8 violations, and review the constrained decoding logic for correctness. All core algorithm design and implementation decisions were made by the author.
