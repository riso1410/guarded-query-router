import tiktoken
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
        "llama3.1:70b": {
            "input_cost_per_token": 0.0000006,
            "output_cost_per_token": 0.00000088,
        },
    }

    costs = model_dict.get(model_name, "gpt-4o-mini")
    prompt_tokens = get_number_of_tokens(prompt)
    completion_tokens = get_number_of_tokens(completion)

    prompt_cost = prompt_tokens * costs["input_cost_per_token"]
    completion_cost = completion_tokens * costs["output_cost_per_token"]
    total_cost = prompt_cost + completion_cost
    return total_cost

def preprocess_data(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        return ' '.join(words)

    # Apply the cleaning function to the 'question' column
    data['question'] = data['question'].apply(clean_text)
    return data