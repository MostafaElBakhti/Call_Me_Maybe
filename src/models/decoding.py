from .validation import Function
import json
from llm_sdk import Small_LLM_Model

def get_best_next_id(logits , valid_vocab):
    """Get the best next token id from valid tokens only.
    
    Args:
        logits: raw scores from the model (151643 scores)
        valid_vocab: set of token ids we allow
    
    Returns:
        the token id with highest score among valid tokens
    """
    best_score = float('-inf')
    best_id = -1

    for token_id in valid_vocab:
        if logits[token_id] > best_score:
            best_score = logits[token_id]
            best_id = token_id

    return best_id


def filter_vocab(id_to_token: dict[int, str]) -> dict[int, str]:
    """Keep only tokens relevant to JSON generation.

    Args:
        id_to_token: Full vocabulary mapping token_id -> token_string.

    Returns:
        Filtered dictionary with only JSON-relevant tokens.
    """
    valid = set(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        '{}":,.-_! '
        '\n'
        '*+?()[]/\\\''  
        'ĠĊ'      
    )

    allowed: dict[int, str] = {}


    for token_id, token_str in id_to_token.items():
        if token_str and all(c in valid for c in token_str):
            allowed[token_id] = token_str

    return allowed



def load_vocabulary(model: Small_LLM_Model):
    """Load the vocabulary from the tokenizer file.
    
    Args:
        model: The loaded Small_LLM_Model instance.
    
    Returns:
        A dictionary mapping token_id (int) -> token_string (str).
    """
    vocab_path = model.get_path_to_tokenizer_file()
    try : 
        with open(vocab_path , "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
            print(list(tokenizer_data.keys()))
            # print(list(tokenizer_data.get("model", {}).keys()))
            print(list(tokenizer_data.get("model", {}).get("vocab", {})))
        vocab = tokenizer_data.get("model", {}).get("vocab", {}) ## we get model then from model we get vocab 
        # print(f"vocav ----> {vocab}")
    except OSError as e:
        raise RuntimeError(f"Failed to read tokenizer file: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in tokenizer file: {e}")
    
    id_to_token = {token_id: token_str for token_str, token_id in vocab.items()}
    return id_to_token


def system_prompt_builder(functions: list[Function]) -> str:
    header = """You are an AI assistant that selects the correct function to call.

        You must ONLY return a JSON object. No explanation. No extra text.

        Rules:
        - Use ONLY the provided functions
        - Use exact function names
        - Use exact parameter names
        - ONLY include parameters listed in the function definition. If the user provides more values than there are parameters, ignore the extra values.
        - Do not add extra fields or extra parameters

        - For regex patterns:
            * "all numbers" or "digits" → use "\\d+"
            * "all vowels" → use "[aeiouAEIOU]"
            * "all letters" → use "[a-zA-Z]"
            * "a specific word" → use the exact word
            * "asterisks" or "star" → use "*"
            * "whitespace" → use "\\s+"
        - For string values containing quotes, use \\" to escape them
            * Example: Say "hello" → "Say \\"hello\\""

        Format:
        {
            "name": "<function_name>",
            "parameters": {
                "<param_name>": <value>
            }
        }

        IMPORTANT: The parameters object must contain ONLY the parameters listed for that function. Nothing more.

        Available functions:
        """
    functions_str = ""
    for fn in functions:
        functions_str += f"\nFunction name: {fn.name}\n"
        functions_str += f"Description: {fn.description}\n"
        functions_str += f"Parameters:\n"

        for name, param in fn.parameters.items():
            functions_str += f"\t- {name} ({param.type})\n"

    return header + functions_str


def get_valid_name_token(generated_token: list , functions_names_ids: dict[str, list[int]]):

    valid_next = set()


    for ids in functions_names_ids.values():
        # ids[:0] li hiya [] == []
        if ids[:len(generated_token)] == generated_token :
            if len(ids) > len(generated_token):

                next_token = ids[len(generated_token)]
                valid_next.add(next_token)

    return valid_next


# ids =  [8822, 5228, 7660, 3904, 6615, 41832], len = 6
# ids =  [8822, 3062, 39794, 12993],
# ids =  [8822, 43277, 3904],
# ids =  [8822, 2891, 32964],
# ids =  [8822, 1889, 3744]