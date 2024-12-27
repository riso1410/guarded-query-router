import tiktoken


def calculate_cost(prompt: str, input: bool) -> float:
    # Calculate and print total cost
    token_count = tiktoken.tokenize(prompt)
    total_tokens = token_count.sum()
    cost_per_1k = 0.00015 if input else 0.0006
    total_cost = (total_tokens / 1000) * cost_per_1k

    return total_cost
