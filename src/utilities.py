import tiktoken


TOKENIZER = tiktoken.encoding_for_model("gpt-4o")

def get_number_of_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def calculate_prompt_cost(
    prompt: str, completion: str, model_name: str = "gpt-4o-mini"
) -> float:
    model_dict = {
        "gpt-4o": {
            "input_cost_per_token": 0.000005,
            "output_cost_per_token": 0.000015,
        },
        "gpt-4o-mini": {
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.00000060,
        },
        # Add llama 
    }

    costs = model_dict.get(model_name, "gpt-4o-mini")
    prompt_tokens = get_number_of_tokens(prompt)
    completion_tokens = get_number_of_tokens(completion)

    prompt_cost = prompt_tokens * costs["input_cost_per_token"]
    completion_cost = completion_tokens * costs["output_cost_per_token"]
    total_cost = prompt_cost + completion_cost
    return total_cost
