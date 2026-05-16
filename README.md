*This project has been created as part of the 42 curriculum by mel-bakh.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling system that maps natural language prompts to structured JSON function calls using a local LLM (Qwen/Qwen3-0.6B). Instead of relying on the model to produce valid JSON through prompting alone, the system uses **constrained decoding** — restricting which tokens the model is allowed to generate at each step — to guarantee that every output is valid, parseable JSON that matches the expected function schema.

The program reads a list of user prompts and a set of function definitions, runs each prompt through the model under token-level constraints, and writes the results to an output JSON file.

## Instructions

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/)

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
  --input data/input/function_calling_tests.json \
  --functions_definition data/input/functions_definition.json \
  --output data/output/function_calling_results.json \
  --model Qwen/Qwen3-0.6B
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/input/function_calling_tests.json` | Path to prompts file |
| `--functions_definition` | `data/input/functions_definition.json` | Path to function schemas |
| `--output` | `data/output/function_calling_results.json` | Path for results |
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace model name |
| `--visualize` | off | Print step-by-step token generation |

**Input format** (`functions_definition.json`):

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

**Input format** (`function_calling_tests.json`):

```json
[
  { "prompt": "What is the sum of 2 and 3?" }
]
```

## Example Usage

```bash
$ uv run python -m src
starting...
loading model ... Qwen/Qwen3-0.6B
Model loaded successfully.
Results saved to data/output/function_calling_results.json
```

**Output** (`function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Algorithm Explanation

The core of the system is **constrained decoding**: at every generation step, instead of letting the model pick any token from its full ~150k vocabulary, we restrict it to only the tokens that are valid given the current position in the JSON structure.

**Step 1 — Input validation**

Both input files are loaded and validated using Pydantic models (`Function`, `Prompt`). Any error — missing file, malformed JSON, wrong schema — raises a `RuntimeError` with a descriptive message and exits cleanly before the model is loaded.

**Step 2 — System prompt construction**

`system_prompt_builder()` builds a prompt listing all available functions with their names, descriptions, and parameter types. It also includes explicit rules for common patterns (e.g. "all numbers" → `\d+`, "all vowels" → `[aeiouAEIOU]`).

**Step 3 — Forced prefix**

For each user prompt, generation begins with the hard-coded prefix `{"name": "`. This forces the model into the correct JSON structure immediately, without any unconstrained generation.

**Step 4 — Constrained function name generation**

`get_valid_name_token()` compares the token IDs generated so far against the pre-encoded token ID sequences of every function name. At each step it returns only the token IDs that are valid continuations of at least one function name. The model's logits are then masked to this set, and the highest-scoring allowed token is selected. Once the generated sequence exactly matches a complete function name, a closing `"` is appended and name generation ends.

**Step 5 — Filtered argument generation**

For argument generation, `filter_vocab()` reduces the vocabulary to tokens composed only of JSON-safe characters (`a–z`, `A–Z`, `0–9`, `{}":,.-_` and a small set of punctuation). This prevents the model from generating prose, unicode, or malformed escape sequences. Generation stops when the output ends with `}}`, with a 200-token safety limit.

**Step 6 — Post-processing**

The generated JSON is parsed, any extra parameters not present in the function definition are removed, and numeric values are cast to `float` or `int` based on the parameter type in the schema. The cleaned result is appended to the output list.

## Design Decisions

**Constrained decoding over pure prompting** — A 0.6B model cannot reliably produce valid JSON through prompting alone. Constraining the token vocabulary at generation time makes correctness a hard guarantee rather than a probabilistic outcome.

**Two-phase generation (name then arguments)** — The name phase uses a prefix-match constraint derived from pre-encoded function name token IDs. The argument phase uses a character-level vocabulary filter. Separating these two phases gives fine-grained control over each part of the output.

**Pydantic for validation** — Input data is validated through Pydantic models before any computation. This catches schema errors early with field-level messages, and makes the data model explicit and type-safe.

**Vocabulary loaded from tokenizer file** — `load_vocabulary()` reads the tokenizer's JSON file via `get_path_to_tokenizer_file()` rather than accessing private model internals. This keeps the implementation within the public SDK interface.

**`get_logits_from_input_ids` for generation** — The main loop works entirely at the token ID level, calling `get_logits_from_input_ids` on the growing sequence instead of encoding/decoding strings at every step. This is the correct level for constrained decoding and avoids string round-trip errors.

## Performance Analysis

Tested on 11 prompts covering simple calls, ambiguous inputs (more values than parameters), and complex regex arguments:

- **Function selection accuracy: 11/11 (100%)**
- **Argument extraction accuracy: 11/11 (100%)**
- **JSON validity: 100%** — every output entry is parseable
- **Processing time:** depends on hardware; the 200-token limit per prompt keeps runtime bounded

The system consistently handles ambiguous prompts (e.g. "add 2 and 3 and 6" with a two-parameter function) by picking the first valid values and ignoring extras, guided by the system prompt rule: *"If the user provides more values than there are parameters, ignore the extra values."*

## Challenges

**Multi-token function names** — Function names like `fn_substitute_string_with_regex` are split into several token IDs by the tokenizer. The prefix-matching logic in `get_valid_name_token()` had to compare full token ID sequences rather than strings, which required pre-encoding every function name once before the generation loop.

**Escape sequences breaking JSON parsing** — The filtered vocabulary occasionally allowed `\d` or `\s` to appear as literal characters in the generated string, which caused `json.loads` to fail. This was resolved by replacing known problematic sequences (`\d` → `\\d`, `\s` → `\\s`) before parsing.

**Extra parameter keys** — The model sometimes generated extra keys in the parameters object. These are removed in post-processing by keeping only the keys that exist in the function definition.

## Testing Strategy

- **Unit-level:** Each error path in `_load_json()` was tested by running the program with malformed JSON, a missing file, an empty array, a JSON object instead of an array, and a schema with wrong keys — all exit cleanly with descriptive messages.
- **Functional:** All 11 prompts in `function_calling_tests.json` were run and the output was manually verified against the expected function names and argument values.
- **Edge cases:** Ambiguous prompts with more inputs than parameters (prompts 1, 3, 5) were verified to produce consistent, reasonable outputs.

## Resources

- [Qwen3 model — HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Constrained decoding — overview](https://huggingface.co/blog/constrained-beam-search)
- [JSON grammar-based generation](https://github.com/outlines-dev/outlines)

**AI usage:** Claude was used during this project for debugging constrained decoding logic, understanding how the tokenizer vocabulary maps to token IDs, and reviewing error handling coverage. All algorithmic decisions and final implementation were written and validated by the student.

## Bonus

**Multiple model support** — the `--model` flag accepts any HuggingFace model
compatible with `llm_sdk`:

```bash
uv run python -m src --model Qwen/Qwen3-0.6B
uv run python -m src --model FrontiersMind/Nandi-Mini-150M-Tool-Calling
uv run python -m src --model FrontiersMind/Nandi-Mini-600M-Early-Checkpoint
```

**Generation visualization** — the token generation process is displayed live
in the terminal, overwriting the same line at each step:

```
[step 0] name so far: fn_
[step 1] name so far: fn_add_numbers
[step 5] generating: {"name": "fn_add_numbers", "parameters": {"a": 2.0
[step 18] generating: {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```